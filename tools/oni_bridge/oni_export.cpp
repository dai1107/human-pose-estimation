#include <OpenNI.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

constexpr const char* kToolVersion = "1.0.0";

struct StreamExport {
    explicit StreamExport(std::string streamName = {})
        : name(std::move(streamName)) {}

    std::string name;
    bool exists = false;
    bool complete = false;
    int width = 0;
    int height = 0;
    int nominal_fps = 0;
    std::string pixel_format = "UNKNOWN";
    std::string frame_encoding = "none";
    int expected_frame_count = 0;
    int actual_frame_count = 0;
    std::int64_t first_timestamp_us = -1;
    std::int64_t last_timestamp_us = -1;
    int first_frame_index = -1;
    int last_frame_index = -1;
    int decode_error_count = 0;
    std::vector<std::string> errors;
};

struct ExportReport {
    std::string input_path_utf8;
    std::string output_path_utf8;
    std::uintmax_t file_size_bytes = 0;
    bool open_success = false;
    bool complete = false;
    std::vector<std::string> errors;
    StreamExport color{"color"};
    StreamExport depth{"depth"};
    StreamExport ir{"ir"};
};

std::string jsonEscape(const std::string& value) {
    std::ostringstream output;
    for (unsigned char character : value) {
        switch (character) {
            case '"':
                output << "\\\"";
                break;
            case '\\':
                output << "\\\\";
                break;
            case '\b':
                output << "\\b";
                break;
            case '\f':
                output << "\\f";
                break;
            case '\n':
                output << "\\n";
                break;
            case '\r':
                output << "\\r";
                break;
            case '\t':
                output << "\\t";
                break;
            default:
                if (character < 0x20) {
                    output << "\\u" << std::hex << std::setw(4)
                           << std::setfill('0')
                           << static_cast<int>(character) << std::dec;
                } else {
                    output << character;
                }
        }
    }
    return output.str();
}

#ifdef _WIN32
std::string wideToUtf8(const std::wstring& value) {
    if (value.empty()) {
        return {};
    }
    const int size = WideCharToMultiByte(
        CP_UTF8, 0, value.c_str(), static_cast<int>(value.size()), nullptr, 0,
        nullptr, nullptr);
    std::string result(static_cast<std::size_t>(size), '\0');
    WideCharToMultiByte(
        CP_UTF8, 0, value.c_str(), static_cast<int>(value.size()),
        result.data(), size, nullptr, nullptr);
    return result;
}

std::wstring absoluteWindowsPath(const std::wstring& value) {
    const DWORD fullSize =
        GetFullPathNameW(value.c_str(), 0, nullptr, nullptr);
    std::wstring fullPath(
        fullSize > 0 ? static_cast<std::size_t>(fullSize) : 0, L'\0');
    if (fullSize > 0) {
        const DWORD written = GetFullPathNameW(
            value.c_str(), fullSize, fullPath.data(), nullptr);
        if (written > 0 && written < fullSize) {
            fullPath.resize(written);
        } else {
            fullPath = value;
        }
    } else {
        fullPath = value;
    }
    return fullPath;
}

bool isAsciiPath(const std::wstring& value) {
    return std::all_of(
        value.begin(), value.end(),
        [](wchar_t character) { return character <= 0x7f; });
}

bool wideToAcpLossless(
    const std::wstring& value,
    std::string& result) {
    BOOL usedDefaultCharacter = FALSE;
    const int size = WideCharToMultiByte(
        CP_ACP, WC_NO_BEST_FIT_CHARS, value.c_str(),
        static_cast<int>(value.size()), nullptr, 0, nullptr,
        &usedDefaultCharacter);
    if (size <= 0 || usedDefaultCharacter) {
        return false;
    }
    result.assign(static_cast<std::size_t>(size), '\0');
    usedDefaultCharacter = FALSE;
    WideCharToMultiByte(
        CP_ACP, WC_NO_BEST_FIT_CHARS, value.c_str(),
        static_cast<int>(value.size()), result.data(), size, nullptr,
        &usedDefaultCharacter);
    return !usedDefaultCharacter;
}

bool removeStagedHardLink(
    const std::wstring& stagedPath,
    const std::wstring& originalPath) {
    if (stagedPath.empty()) {
        return true;
    }
    HANDLE handle = CreateFileW(
        stagedPath.c_str(), DELETE,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle != INVALID_HANDLE_VALUE) {
        struct FileDispositionInfoExCompat {
            DWORD flags;
        };
        constexpr DWORD kDelete = 0x00000001;
        constexpr DWORD kPosixSemantics = 0x00000002;
        constexpr DWORD kIgnoreReadonlyAttribute = 0x00000010;
        FileDispositionInfoExCompat disposition{
            kDelete | kPosixSemantics | kIgnoreReadonlyAttribute};
        constexpr int kFileDispositionInfoEx = 21;
        const BOOL removed = SetFileInformationByHandle(
            handle,
            static_cast<FILE_INFO_BY_HANDLE_CLASS>(
                kFileDispositionInfoEx),
            &disposition,
            sizeof(disposition));
        CloseHandle(handle);
        if (removed) {
            return true;
        }
    }
    const DWORD originalAttributes =
        GetFileAttributesW(originalPath.c_str());
    const DWORD stagedAttributes =
        GetFileAttributesW(stagedPath.c_str());
    if (stagedAttributes == INVALID_FILE_ATTRIBUTES) {
        return true;
    }
    if (!SetFileAttributesW(
            stagedPath.c_str(),
            stagedAttributes & ~FILE_ATTRIBUTE_READONLY)) {
        return false;
    }
    const BOOL removed = DeleteFileW(stagedPath.c_str());
    if (originalAttributes != INVALID_FILE_ATTRIBUTES &&
        GetFileAttributesW(originalPath.c_str()) !=
            INVALID_FILE_ATTRIBUTES) {
        SetFileAttributesW(originalPath.c_str(), originalAttributes);
    }
    return removed != FALSE;
}

std::string prepareOpenNiPath(
    const std::wstring& value,
    std::wstring& stagedPath,
    std::string& error) {
    const std::wstring fullPath = absoluteWindowsPath(value);
    std::string result;
    if (isAsciiPath(fullPath) && wideToAcpLossless(fullPath, result)) {
        return result;
    }
    const DWORD shortSize = GetShortPathNameW(fullPath.c_str(), nullptr, 0);
    if (shortSize > 0) {
        std::wstring shortPath(
            static_cast<std::size_t>(shortSize), L'\0');
        const DWORD written = GetShortPathNameW(
            fullPath.c_str(), shortPath.data(), shortSize);
        if (written > 0 && written < shortSize) {
            shortPath.resize(written);
            if (isAsciiPath(shortPath) &&
                wideToAcpLossless(shortPath, result)) {
                return result;
            }
        }
    }
    wchar_t temporaryRoot[MAX_PATH + 1] = {};
    const DWORD temporaryRootLength =
        GetTempPathW(MAX_PATH, temporaryRoot);
    if (temporaryRootLength == 0 ||
        temporaryRootLength > MAX_PATH) {
        error = "GetTempPathW_error_" +
                std::to_string(
                    static_cast<unsigned long>(GetLastError()));
        return {};
    }
    for (unsigned int attempt = 0; attempt < 100; ++attempt) {
        std::wostringstream candidate;
        candidate << temporaryRoot << L"oni_export_"
                  << GetCurrentProcessId() << L"_"
                  << GetTickCount() << L"_" << attempt << L".oni";
        stagedPath = candidate.str();
        if (CreateHardLinkW(
                stagedPath.c_str(), fullPath.c_str(), nullptr)) {
            if (isAsciiPath(stagedPath) &&
                wideToAcpLossless(stagedPath, result)) {
                return result;
            }
            removeStagedHardLink(stagedPath, fullPath);
            stagedPath.clear();
            error = "temporary_hard_link_path_not_ascii";
            return {};
        }
        const DWORD createError = GetLastError();
        if (createError != ERROR_FILE_EXISTS &&
            createError != ERROR_ALREADY_EXISTS) {
            error = "CreateHardLinkW_error_" +
                    std::to_string(
                        static_cast<unsigned long>(createError));
            stagedPath.clear();
            return {};
        }
    }
    error = "temporary_hard_link_name_exhausted";
    return {};
}

bool getFileSize(
    const std::wstring& path,
    std::uintmax_t& size,
    std::string& error) {
    WIN32_FILE_ATTRIBUTE_DATA attributes{};
    if (!GetFileAttributesExW(
            path.c_str(), GetFileExInfoStandard, &attributes)) {
        error = "GetFileAttributesExW_error_" +
                std::to_string(
                    static_cast<unsigned long>(GetLastError()));
        return false;
    }
    size =
        (static_cast<std::uintmax_t>(attributes.nFileSizeHigh) << 32U) |
        static_cast<std::uintmax_t>(attributes.nFileSizeLow);
    return true;
}

bool createDirectoryRecursive(const std::wstring& rawPath) {
    if (rawPath.empty()) {
        return false;
    }
    std::wstring path = absoluteWindowsPath(rawPath);
    std::replace(path.begin(), path.end(), L'/', L'\\');
    std::size_t position = path.size();
    while (position > 0 &&
           (path[position - 1] == L'\\' || path[position - 1] == L'/')) {
        --position;
    }
    path.resize(position);
    for (std::size_t index = 3; index <= path.size(); ++index) {
        if (index == path.size() || path[index] == L'\\') {
            const std::wstring part = path.substr(0, index);
            if (!CreateDirectoryW(part.c_str(), nullptr)) {
                const DWORD createError = GetLastError();
                if (createError != ERROR_ALREADY_EXISTS) {
                    return false;
                }
            }
        }
    }
    return true;
}

std::FILE* openWideFile(
    const std::wstring& path,
    const wchar_t* mode) {
    return _wfopen(path.c_str(), mode);
}
#else
std::string wideToUtf8(const std::wstring& value) {
    return std::string(value.begin(), value.end());
}

std::string prepareOpenNiPath(
    const std::wstring& value,
    std::wstring&,
    std::string&) {
    return wideToUtf8(value);
}

bool getFileSize(
    const std::wstring& path,
    std::uintmax_t& size,
    std::string& error) {
    std::FILE* file = std::fopen(wideToUtf8(path).c_str(), "rb");
    if (file == nullptr) {
        error = "fopen_failed";
        return false;
    }
    std::fseek(file, 0, SEEK_END);
    const long position = std::ftell(file);
    std::fclose(file);
    if (position < 0) {
        error = "ftell_failed";
        return false;
    }
    size = static_cast<std::uintmax_t>(position);
    return true;
}

bool createDirectoryRecursive(const std::wstring&) {
    return false;
}

std::FILE* openWideFile(
    const std::wstring& path,
    const wchar_t*) {
    return std::fopen(wideToUtf8(path).c_str(), "wb");
}
#endif

std::wstring joinPath(
    const std::wstring& parent,
    const std::wstring& child) {
    if (parent.empty()) {
        return child;
    }
    if (parent.back() == L'\\' || parent.back() == L'/') {
        return parent + child;
    }
    return parent + L"\\" + child;
}

std::string pixelFormatName(openni::PixelFormat format) {
    switch (format) {
        case openni::PIXEL_FORMAT_DEPTH_1_MM:
            return "DEPTH_1_MM";
        case openni::PIXEL_FORMAT_DEPTH_100_UM:
            return "DEPTH_100_UM";
        case openni::PIXEL_FORMAT_SHIFT_9_2:
            return "SHIFT_9_2";
        case openni::PIXEL_FORMAT_SHIFT_9_3:
            return "SHIFT_9_3";
        case openni::PIXEL_FORMAT_RGB888:
            return "RGB888";
        case openni::PIXEL_FORMAT_YUV422:
            return "YUV422";
        case openni::PIXEL_FORMAT_GRAY8:
            return "GRAY8";
        case openni::PIXEL_FORMAT_GRAY16:
            return "GRAY16";
        case openni::PIXEL_FORMAT_JPEG:
            return "JPEG";
        default:
            return "UNKNOWN_" +
                   std::to_string(static_cast<int>(format));
    }
}

openni::SensorType sensorTypeForName(const std::string& name) {
    if (name == "color") {
        return openni::SENSOR_COLOR;
    }
    if (name == "depth") {
        return openni::SENSOR_DEPTH;
    }
    return openni::SENSOR_IR;
}

std::wstring frameStem(int outputFrame) {
    std::wostringstream output;
    output << std::setw(8) << std::setfill(L'0') << outputFrame;
    return output.str();
}

void writeAll(
    std::FILE* file,
    const void* data,
    std::size_t size,
    const std::string& context) {
    if (size == 0) {
        return;
    }
    if (std::fwrite(data, 1, size, file) != size) {
        throw std::runtime_error("write failed: " + context);
    }
}

void writeNpyHeader(
    std::FILE* file,
    const std::string& descriptor,
    const std::vector<int>& shape) {
    std::ostringstream shapeText;
    shapeText << "(";
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index > 0) {
            shapeText << ", ";
        }
        shapeText << shape[index];
    }
    if (shape.size() == 1) {
        shapeText << ",";
    }
    shapeText << ")";
    std::string header =
        "{'descr': '" + descriptor +
        "', 'fortran_order': False, 'shape': " +
        shapeText.str() + ", }";
    while ((10 + header.size() + 1) % 16 != 0) {
        header.push_back(' ');
    }
    header.push_back('\n');
    if (header.size() > 65535) {
        throw std::runtime_error("NPY header too large");
    }
    const unsigned char magic[] = {
        0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    writeAll(file, magic, sizeof(magic), "NPY magic");
    const std::uint16_t headerLength =
        static_cast<std::uint16_t>(header.size());
    const unsigned char lengthBytes[] = {
        static_cast<unsigned char>(headerLength & 0xff),
        static_cast<unsigned char>((headerLength >> 8) & 0xff)};
    writeAll(file, lengthBytes, sizeof(lengthBytes), "NPY header length");
    writeAll(file, header.data(), header.size(), "NPY header");
}

void writeNpyRows(
    const std::wstring& path,
    const openni::VideoFrameRef& frame,
    int bytesPerPixel,
    const std::string& descriptor,
    const std::vector<int>& shape) {
    std::FILE* output = openWideFile(path, L"wb");
    if (output == nullptr) {
        throw std::runtime_error(
            "cannot open frame output: " + wideToUtf8(path));
    }
    try {
        writeNpyHeader(output, descriptor, shape);
        const auto* base =
            static_cast<const std::uint8_t*>(frame.getData());
        const int rowBytes = frame.getWidth() * bytesPerPixel;
        for (int row = 0; row < frame.getHeight(); ++row) {
            writeAll(
                output,
                base + row * frame.getStrideInBytes(),
                static_cast<std::size_t>(rowBytes),
                wideToUtf8(path));
        }
    } catch (...) {
        std::fclose(output);
        throw;
    }
    if (std::fclose(output) != 0) {
        throw std::runtime_error(
            "cannot close frame output: " + wideToUtf8(path));
    }
}

void writePortableImage(
    const std::wstring& path,
    const openni::VideoFrameRef& frame,
    int channels) {
    std::FILE* output = openWideFile(path, L"wb");
    if (output == nullptr) {
        throw std::runtime_error(
            "cannot open frame output: " + wideToUtf8(path));
    }
    std::ostringstream header;
    header << (channels == 3 ? "P6\n" : "P5\n")
           << frame.getWidth() << " " << frame.getHeight() << "\n255\n";
    const std::string headerBytes = header.str();
    try {
        writeAll(
            output, headerBytes.data(), headerBytes.size(),
            wideToUtf8(path));
        const auto* base =
            static_cast<const std::uint8_t*>(frame.getData());
        const int rowBytes = frame.getWidth() * channels;
        for (int row = 0; row < frame.getHeight(); ++row) {
            writeAll(
                output,
                base + row * frame.getStrideInBytes(),
                static_cast<std::size_t>(rowBytes),
                wideToUtf8(path));
        }
    } catch (...) {
        std::fclose(output);
        throw;
    }
    if (std::fclose(output) != 0) {
        throw std::runtime_error(
            "cannot close frame output: " + wideToUtf8(path));
    }
}

void writeRawFrame(
    const std::wstring& path,
    const openni::VideoFrameRef& frame) {
    std::FILE* output = openWideFile(path, L"wb");
    if (output == nullptr) {
        throw std::runtime_error(
            "cannot open frame output: " + wideToUtf8(path));
    }
    writeAll(
        output, frame.getData(),
        static_cast<std::size_t>(frame.getDataSize()),
        wideToUtf8(path));
    if (std::fclose(output) != 0) {
        throw std::runtime_error(
            "cannot close frame output: " + wideToUtf8(path));
    }
}

double invalidDepthRatio(const openni::VideoFrameRef& frame) {
    const auto format = frame.getVideoMode().getPixelFormat();
    if (format != openni::PIXEL_FORMAT_DEPTH_1_MM &&
        format != openni::PIXEL_FORMAT_DEPTH_100_UM) {
        return 0.0;
    }
    std::uint64_t total = 0;
    std::uint64_t invalid = 0;
    const auto* base =
        static_cast<const std::uint8_t*>(frame.getData());
    for (int row = 0; row < frame.getHeight(); ++row) {
        const auto* values = reinterpret_cast<const std::uint16_t*>(
            base + row * frame.getStrideInBytes());
        for (int column = 0; column < frame.getWidth(); ++column) {
            ++total;
            if (values[column] == 0) {
                ++invalid;
            }
        }
    }
    return total == 0
               ? 0.0
               : static_cast<double>(invalid) /
                     static_cast<double>(total);
}

std::string exportFrame(
    const std::wstring& framesRoot,
    const std::string& streamName,
    int outputFrame,
    const openni::VideoFrameRef& frame) {
    const auto format = frame.getVideoMode().getPixelFormat();
    const std::wstring stem = frameStem(outputFrame);
    if (streamName == "depth") {
        const std::wstring path = joinPath(framesRoot, stem + L".npy");
        writeNpyRows(
            path, frame, 2, "<u2",
            {frame.getHeight(), frame.getWidth()});
        return "npy_uint16_little_endian";
    }
    if (streamName == "ir" &&
        format == openni::PIXEL_FORMAT_GRAY16) {
        const std::wstring path = joinPath(framesRoot, stem + L".npy");
        writeNpyRows(
            path, frame, 2, "<u2",
            {frame.getHeight(), frame.getWidth()});
        return "npy_uint16_little_endian";
    }
    if (streamName == "ir" &&
        format == openni::PIXEL_FORMAT_GRAY8) {
        const std::wstring path = joinPath(framesRoot, stem + L".pgm");
        writePortableImage(path, frame, 1);
        return "pgm_gray8";
    }
    if (streamName == "color" &&
        format == openni::PIXEL_FORMAT_RGB888) {
        const std::wstring path = joinPath(framesRoot, stem + L".ppm");
        writePortableImage(path, frame, 3);
        return "ppm_rgb888";
    }
    if (streamName == "color" &&
        format == openni::PIXEL_FORMAT_JPEG) {
        const std::wstring path = joinPath(framesRoot, stem + L".jpg");
        writeRawFrame(path, frame);
        return "jpeg_original_bytes";
    }
    const std::wstring path = joinPath(framesRoot, stem + L".bin");
    writeRawFrame(path, frame);
    return "raw_frame_bytes";
}

void writeStringArray(
    std::ostream& output,
    const std::vector<std::string>& values) {
    output << "[";
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index > 0) {
            output << ", ";
        }
        output << "\"" << jsonEscape(values[index]) << "\"";
    }
    output << "]";
}

StreamExport exportStream(
    const std::string& inputPath,
    const std::wstring& outputRoot,
    const std::string& streamName) {
    StreamExport result(streamName);
    const std::wstring streamRoot =
        joinPath(outputRoot, std::wstring(streamName.begin(), streamName.end()));
    const std::wstring framesRoot = joinPath(streamRoot, L"frames");
    if (!createDirectoryRecursive(framesRoot)) {
        result.errors.push_back("cannot_create_output_directories");
        return result;
    }
    const std::wstring indexPath = joinPath(streamRoot, L"index.csv");
    std::FILE* index = openWideFile(indexPath, L"wb");
    if (index == nullptr) {
        result.errors.push_back("cannot_open_index_csv");
        return result;
    }
    const bool isDepth = streamName == "depth";
    const char* header =
        isDepth
            ? "output_frame,source_frame_index,timestamp_us,depth_scale,"
              "invalid_pixel_ratio\n"
            : "output_frame,source_frame_index,timestamp_us,width,height,"
              "pixel_format\n";
    writeAll(index, header, std::strlen(header), "index header");

    openni::Device device;
    openni::Status status = device.open(inputPath.c_str());
    if (status != openni::STATUS_OK) {
        result.errors.push_back(
            std::string("device_open_failed: ") +
            openni::OpenNI::getExtendedError());
        std::fclose(index);
        return result;
    }
    const openni::SensorType sensorType =
        sensorTypeForName(streamName);
    result.exists = device.hasSensor(sensorType);
    if (!result.exists) {
        device.close();
        std::fclose(index);
        return result;
    }
    openni::VideoStream stream;
    status = stream.create(device, sensorType);
    if (status != openni::STATUS_OK) {
        result.errors.push_back(
            std::string("stream_create_failed: ") +
            openni::OpenNI::getExtendedError());
        device.close();
        std::fclose(index);
        return result;
    }
    const openni::VideoMode mode = stream.getVideoMode();
    result.width = mode.getResolutionX();
    result.height = mode.getResolutionY();
    result.nominal_fps = mode.getFps();
    result.pixel_format = pixelFormatName(mode.getPixelFormat());
    openni::PlaybackControl* playback = device.getPlaybackControl();
    if (playback == nullptr) {
        result.errors.push_back("playback_control_unavailable");
        stream.destroy();
        device.close();
        std::fclose(index);
        return result;
    }
    playback->setRepeatEnabled(false);
    playback->setSpeed(-1.0f);
    result.expected_frame_count = playback->getNumberOfFrames(stream);
    if (result.expected_frame_count < 0) {
        result.expected_frame_count = 0;
    }
    status = stream.start();
    if (status != openni::STATUS_OK) {
        result.errors.push_back(
            std::string("stream_start_failed: ") +
            openni::OpenNI::getExtendedError());
        stream.destroy();
        device.close();
        std::fclose(index);
        return result;
    }

    const int hardFrameLimit =
        result.expected_frame_count > 0
            ? result.expected_frame_count
            : 10'000'000;
    try {
        for (int outputFrame = 0;
             outputFrame < hardFrameLimit;
             ++outputFrame) {
            openni::VideoFrameRef frame;
            status = stream.readFrame(&frame);
            if (status != openni::STATUS_OK || !frame.isValid()) {
                ++result.decode_error_count;
                result.errors.push_back(
                    std::string("frame_read_failed_after_") +
                    std::to_string(result.actual_frame_count) +
                    "_frames: " +
                    openni::OpenNI::getExtendedError());
                break;
            }
            if (result.actual_frame_count == 0) {
                result.first_timestamp_us =
                    static_cast<std::int64_t>(frame.getTimestamp());
                result.first_frame_index = frame.getFrameIndex();
            }
            result.last_timestamp_us =
                static_cast<std::int64_t>(frame.getTimestamp());
            result.last_frame_index = frame.getFrameIndex();
            const std::string encoding = exportFrame(
                framesRoot, streamName, outputFrame, frame);
            if (result.frame_encoding == "none") {
                result.frame_encoding = encoding;
            } else if (result.frame_encoding != encoding) {
                result.frame_encoding = "mixed";
            }
            std::ostringstream row;
            row << std::fixed << std::setprecision(9);
            if (isDepth) {
                const double depthScale =
                    mode.getPixelFormat() ==
                            openni::PIXEL_FORMAT_DEPTH_100_UM
                        ? 0.1
                        : 1.0;
                row << outputFrame << ","
                    << frame.getFrameIndex() << ","
                    << frame.getTimestamp() << ","
                    << depthScale << ","
                    << invalidDepthRatio(frame) << "\n";
            } else {
                row << outputFrame << ","
                    << frame.getFrameIndex() << ","
                    << frame.getTimestamp() << ","
                    << frame.getWidth() << ","
                    << frame.getHeight() << ","
                    << pixelFormatName(
                           frame.getVideoMode().getPixelFormat())
                    << "\n";
            }
            const std::string rowBytes = row.str();
            writeAll(
                index, rowBytes.data(), rowBytes.size(),
                wideToUtf8(indexPath));
            ++result.actual_frame_count;
            if (result.expected_frame_count > 0 &&
                result.actual_frame_count >=
                    result.expected_frame_count) {
                break;
            }
        }
    } catch (const std::exception& error) {
        result.errors.push_back(
            std::string("frame_export_failed: ") + error.what());
    }

    stream.stop();
    stream.destroy();
    device.close();
    if (std::fclose(index) != 0) {
        result.errors.push_back("cannot_close_index_csv");
    }
    result.complete =
        result.decode_error_count == 0 &&
        result.actual_frame_count > 0 &&
        (result.expected_frame_count == 0 ||
         result.actual_frame_count == result.expected_frame_count) &&
        result.errors.empty();
    return result;
}

void writeStreamJson(
    std::ostream& output,
    const StreamExport& stream) {
    output << "{\n";
    output << "      \"exists\": "
           << (stream.exists ? "true" : "false") << ",\n";
    output << "      \"complete\": "
           << (stream.complete ? "true" : "false") << ",\n";
    output << "      \"width\": " << stream.width << ",\n";
    output << "      \"height\": " << stream.height << ",\n";
    output << "      \"pixel_format\": \""
           << jsonEscape(stream.pixel_format) << "\",\n";
    output << "      \"nominal_fps\": "
           << stream.nominal_fps << ",\n";
    output << "      \"frame_encoding\": \""
           << jsonEscape(stream.frame_encoding) << "\",\n";
    output << "      \"expected_frame_count\": "
           << stream.expected_frame_count << ",\n";
    output << "      \"actual_frame_count\": "
           << stream.actual_frame_count << ",\n";
    output << "      \"first_timestamp_us\": "
           << (stream.first_timestamp_us < 0
                   ? "null"
                   : std::to_string(stream.first_timestamp_us))
           << ",\n";
    output << "      \"last_timestamp_us\": "
           << (stream.last_timestamp_us < 0
                   ? "null"
                   : std::to_string(stream.last_timestamp_us))
           << ",\n";
    output << "      \"first_frame_index\": "
           << (stream.first_frame_index < 0
                   ? "null"
                   : std::to_string(stream.first_frame_index))
           << ",\n";
    output << "      \"last_frame_index\": "
           << (stream.last_frame_index < 0
                   ? "null"
                   : std::to_string(stream.last_frame_index))
           << ",\n";
    output << "      \"decode_error_count\": "
           << stream.decode_error_count << ",\n";
    output << "      \"errors\": ";
    writeStringArray(output, stream.errors);
    output << "\n    }";
}

void writeMetadata(
    const std::wstring& outputPath,
    const ExportReport& report,
    const openni::Version& version) {
    std::ostringstream output;
    output << "{\n";
    output << "  \"schema_version\": 1,\n";
    output << "  \"artifact_type\": \"oni_lossless_export\",\n";
    output << "  \"tool\": {\n";
    output << "    \"name\": \"oni-export\",\n";
    output << "    \"version\": \"" << kToolVersion << "\",\n";
    output << "    \"openni_version\": \""
           << version.major << "." << version.minor << "."
           << version.maintenance << "." << version.build << "\",\n";
    output << "    \"offline_file_only\": true\n";
    output << "  },\n";
    output << "  \"input\": {\n";
    output << "    \"path\": \""
           << jsonEscape(report.input_path_utf8) << "\",\n";
    output << "    \"size_bytes\": "
           << report.file_size_bytes << ",\n";
    output << "    \"open_success\": "
           << (report.open_success ? "true" : "false") << "\n";
    output << "  },\n";
    output << "  \"output_path\": \""
           << jsonEscape(report.output_path_utf8) << "\",\n";
    output << "  \"complete\": "
           << (report.complete ? "true" : "false") << ",\n";
    output << "  \"lossless_depth\": true,\n";
    output << "  \"playback_speed_independent\": true,\n";
    output << "  \"errors\": ";
    writeStringArray(output, report.errors);
    output << ",\n";
    output << "  \"streams\": {\n";
    output << "    \"color\": ";
    writeStreamJson(output, report.color);
    output << ",\n    \"depth\": ";
    writeStreamJson(output, report.depth);
    output << ",\n    \"ir\": ";
    writeStreamJson(output, report.ir);
    output << "\n  }\n";
    output << "}\n";
    const std::string serialized = output.str();
    std::FILE* file = openWideFile(outputPath, L"wb");
    if (file == nullptr) {
        throw std::runtime_error(
            "cannot open metadata output: " +
            wideToUtf8(outputPath));
    }
    writeAll(
        file, serialized.data(), serialized.size(),
        wideToUtf8(outputPath));
    if (std::fclose(file) != 0) {
        throw std::runtime_error(
            "cannot close metadata output: " +
            wideToUtf8(outputPath));
    }
}

int exportOni(
    const std::wstring& inputWide,
    const std::wstring& outputWide) {
    ExportReport report;
    report.input_path_utf8 = wideToUtf8(inputWide);
    report.output_path_utf8 = wideToUtf8(outputWide);
    if (!createDirectoryRecursive(outputWide) ||
        !createDirectoryRecursive(joinPath(outputWide, L"preview"))) {
        throw std::runtime_error("cannot create export output directory");
    }
    std::string fileSizeError;
    if (!getFileSize(
            inputWide, report.file_size_bytes, fileSizeError)) {
        report.errors.push_back(
            "file_size_failed: " + fileSizeError);
    }
    const openni::Status initialization =
        openni::OpenNI::initialize();
    const openni::Version version = openni::OpenNI::getVersion();
    const std::wstring metadataPath =
        joinPath(outputWide, L"metadata.json");
    if (initialization != openni::STATUS_OK) {
        report.errors.push_back(
            std::string("openni_initialize_failed: ") +
            openni::OpenNI::getExtendedError());
        writeMetadata(metadataPath, report, version);
        return 3;
    }
    std::wstring stagedInputPath;
    std::string pathError;
    const std::string inputPath = prepareOpenNiPath(
        inputWide, stagedInputPath, pathError);
    if (inputPath.empty()) {
        report.errors.push_back(
            "openni_path_prepare_failed: " + pathError);
        writeMetadata(metadataPath, report, version);
        openni::OpenNI::shutdown();
        return 3;
    }
    {
        openni::Device device;
        report.open_success =
            device.open(inputPath.c_str()) == openni::STATUS_OK;
        if (!report.open_success) {
            report.errors.push_back(
                std::string("device_open_failed: ") +
                openni::OpenNI::getExtendedError());
        }
        device.close();
    }
    if (report.open_success) {
        report.color =
            exportStream(inputPath, outputWide, "color");
        report.depth =
            exportStream(inputPath, outputWide, "depth");
        report.ir =
            exportStream(inputPath, outputWide, "ir");
        const bool anyStream =
            report.color.exists ||
            report.depth.exists ||
            report.ir.exists;
        report.complete =
            anyStream &&
            (!report.color.exists || report.color.complete) &&
            (!report.depth.exists || report.depth.complete) &&
            (!report.ir.exists || report.ir.complete);
    }
    writeMetadata(metadataPath, report, version);
    openni::OpenNI::shutdown();
#ifdef _WIN32
    if (!stagedInputPath.empty() &&
        !removeStagedHardLink(stagedInputPath, inputWide)) {
        std::cerr
            << "warning: could not remove temporary ONI hard link\n";
    }
#endif
    return report.complete ? 0 : 3;
}

void printUsage() {
    std::wcerr
        << L"Usage: oni-export.exe input.oni --output extracted/record_id\n";
}

}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
    if (argc != 4 || std::wstring(argv[2]) != L"--output") {
        printUsage();
        return 2;
    }
    try {
        return exportOni(argv[1], argv[3]);
    } catch (const std::exception& error) {
        std::cerr << "oni-export fatal error: "
                  << error.what() << "\n";
        return 4;
    }
}
#else
int main(int argc, char* argv[]) {
    if (argc != 4 || std::string(argv[2]) != "--output") {
        printUsage();
        return 2;
    }
    try {
        return exportOni(
            std::wstring(argv[1], argv[1] + std::strlen(argv[1])),
            std::wstring(argv[3], argv[3] + std::strlen(argv[3])));
    } catch (const std::exception& error) {
        std::cerr << "oni-export fatal error: "
                  << error.what() << "\n";
        return 4;
    }
}
#endif
