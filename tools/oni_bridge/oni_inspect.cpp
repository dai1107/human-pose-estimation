#include <OpenNI.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
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

struct DepthQuality {
    std::uint64_t total_pixels = 0;
    std::uint64_t zero_pixels = 0;
    std::uint64_t valid_pixels = 0;
    std::uint16_t min_depth = std::numeric_limits<std::uint16_t>::max();
    std::uint16_t max_depth = 0;
    std::uint64_t center_total_pixels = 0;
    std::uint64_t center_zero_pixels = 0;
    std::uint64_t center_valid_pixels = 0;
    std::uint16_t center_min_depth = std::numeric_limits<std::uint16_t>::max();
    std::uint16_t center_max_depth = 0;
    std::vector<std::uint64_t> center_histogram =
        std::vector<std::uint64_t>(65536, 0);
    double depth_scale_mm = 0.0;
};

struct StreamAudit {
    explicit StreamAudit(std::string streamName = {})
        : name(std::move(streamName)) {}

    std::string name;
    bool exists = false;
    bool create_success = false;
    bool start_success = false;
    bool complete = false;
    int width = 0;
    int height = 0;
    int nominal_fps = 0;
    std::string pixel_format = "UNKNOWN";
    int expected_frame_count = 0;
    int actual_frame_count = 0;
    std::int64_t first_timestamp_us = -1;
    std::int64_t last_timestamp_us = -1;
    int first_frame_index = -1;
    int last_frame_index = -1;
    bool timestamps_strictly_increasing = true;
    bool frame_indices_continuous = true;
    int non_increasing_timestamp_count = 0;
    int frame_index_discontinuity_count = 0;
    int estimated_dropped_frames = 0;
    int abnormal_interval_count = 0;
    int decode_error_count = 0;
    double interval_p50_us = 0.0;
    double interval_p95_us = 0.0;
    double actual_fps = 0.0;
    std::vector<std::int64_t> positive_intervals_us;
    std::vector<std::string> errors;
    DepthQuality depth;
};

struct FileAudit {
    std::string input_path_utf8;
    std::uintmax_t file_size_bytes = 0;
    bool open_success = false;
    bool complete_playback = false;
    std::int64_t duration_us = 0;
    std::string classification = "E";
    std::string classification_description = "corrupt_or_unreadable";
    bool qualified_for_rgbd = false;
    std::vector<std::string> errors;
    StreamAudit color{"color"};
    StreamAudit depth{"depth"};
    StreamAudit ir{"ir"};
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
                           << std::setfill('0') << static_cast<int>(character)
                           << std::dec;
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
        CP_UTF8, 0, value.c_str(), static_cast<int>(value.size()), result.data(),
        size, nullptr, nullptr);
    return result;
}

std::wstring absoluteWindowsPath(const std::wstring& value) {
    const DWORD fullSize = GetFullPathNameW(
        value.c_str(), 0, nullptr, nullptr);
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

bool isAsciiPath(const std::wstring& value) {
    return std::all_of(
        value.begin(), value.end(),
        [](wchar_t character) {
            return character <= 0x7f;
        });
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

std::string wideToOpenNiPath(
    const std::wstring& value,
    std::wstring& stagedPath,
    std::string& error) {
    if (value.empty()) {
        error = "empty_input_path";
        return {};
    }
    const std::wstring fullPath = absoluteWindowsPath(value);
    std::string result;
    if (isAsciiPath(fullPath) &&
        wideToAcpLossless(fullPath, result)) {
        return result;
    }

    const DWORD shortSize = GetShortPathNameW(
        fullPath.c_str(), nullptr, 0);
    if (shortSize > 0) {
        std::wstring shortPath(
            static_cast<std::size_t>(shortSize), L'\0');
        const DWORD shortWritten = GetShortPathNameW(
            fullPath.c_str(), shortPath.data(), shortSize);
        if (shortWritten > 0 && shortWritten < shortSize) {
            shortPath.resize(shortWritten);
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
        error =
            "GetTempPathW_error_" +
            std::to_string(static_cast<unsigned long>(GetLastError()));
        return {};
    }
    for (unsigned int attempt = 0; attempt < 100; ++attempt) {
        std::wostringstream candidate;
        candidate << temporaryRoot << L"oni_inspect_"
                  << GetCurrentProcessId() << L"_"
                  << GetTickCount() << L"_" << attempt << L".oni";
        stagedPath = candidate.str();
        if (CreateHardLinkW(
                stagedPath.c_str(), fullPath.c_str(), nullptr)) {
            if (wideToAcpLossless(stagedPath, result)) {
                return result;
            }
            DeleteFileW(stagedPath.c_str());
            stagedPath.clear();
            error = "temporary_hard_link_path_not_acp_representable";
            return {};
        }
        if (GetLastError() != ERROR_FILE_EXISTS &&
            GetLastError() != ERROR_ALREADY_EXISTS) {
            error =
                "CreateHardLinkW_error_" +
                std::to_string(
                    static_cast<unsigned long>(GetLastError()));
            stagedPath.clear();
            return {};
        }
    }
    error = "temporary_hard_link_name_exhausted";
    stagedPath.clear();
    return {};
}

bool getFileSize(
    const std::wstring& path,
    std::uintmax_t& size,
    std::string& error) {
    WIN32_FILE_ATTRIBUTE_DATA attributes{};
    if (!GetFileAttributesExW(
            path.c_str(), GetFileExInfoStandard, &attributes)) {
        error =
            "GetFileAttributesExW_error_" +
            std::to_string(static_cast<unsigned long>(GetLastError()));
        return false;
    }
    size =
        (static_cast<std::uintmax_t>(attributes.nFileSizeHigh) << 32U) |
        static_cast<std::uintmax_t>(attributes.nFileSizeLow);
    return true;
}
#else
std::string wideToUtf8(const std::wstring& value) {
    return std::string(value.begin(), value.end());
}

std::string wideToOpenNiPath(
    const std::wstring& value,
    std::wstring&,
    std::string&) {
    return wideToUtf8(value);
}

bool getFileSize(
    const std::wstring& path,
    std::uintmax_t& size,
    std::string& error) {
    const std::string narrowPath = wideToUtf8(path);
    std::FILE* file = std::fopen(narrowPath.c_str(), "rb");
    if (file == nullptr) {
        error = "fopen_failed";
        return false;
    }
    if (std::fseek(file, 0, SEEK_END) != 0) {
        std::fclose(file);
        error = "fseek_failed";
        return false;
    }
    const long position = std::ftell(file);
    std::fclose(file);
    if (position < 0) {
        error = "ftell_failed";
        return false;
    }
    size = static_cast<std::uintmax_t>(position);
    return true;
}
#endif

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
            return "UNKNOWN_" + std::to_string(static_cast<int>(format));
    }
}

double percentile(std::vector<std::int64_t> values, double percentileValue) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double rank =
        (static_cast<double>(values.size()) - 1.0) * percentileValue / 100.0;
    const std::size_t lower = static_cast<std::size_t>(std::floor(rank));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(rank));
    if (lower == upper) {
        return static_cast<double>(values[lower]);
    }
    const double fraction = rank - static_cast<double>(lower);
    return static_cast<double>(values[lower]) * (1.0 - fraction) +
           static_cast<double>(values[upper]) * fraction;
}

std::uint16_t histogramPercentile(
    const std::vector<std::uint64_t>& histogram,
    std::uint64_t count,
    double percentileValue) {
    if (count == 0) {
        return 0;
    }
    const std::uint64_t target = static_cast<std::uint64_t>(
        std::ceil(static_cast<double>(count) * percentileValue / 100.0));
    std::uint64_t cumulative = 0;
    for (std::size_t value = 1; value < histogram.size(); ++value) {
        cumulative += histogram[value];
        if (cumulative >= std::max<std::uint64_t>(target, 1)) {
            return static_cast<std::uint16_t>(value);
        }
    }
    return 0;
}

void updateDepthQuality(
    const openni::VideoFrameRef& frame,
    DepthQuality& quality) {
    const openni::PixelFormat format = frame.getVideoMode().getPixelFormat();
    if (format != openni::PIXEL_FORMAT_DEPTH_1_MM &&
        format != openni::PIXEL_FORMAT_DEPTH_100_UM) {
        return;
    }
    quality.depth_scale_mm =
        format == openni::PIXEL_FORMAT_DEPTH_100_UM ? 0.1 : 1.0;
    const int width = frame.getWidth();
    const int height = frame.getHeight();
    const int stride = frame.getStrideInBytes();
    const auto* base = static_cast<const std::uint8_t*>(frame.getData());
    const int centerX0 = width / 4;
    const int centerX1 = width - centerX0;
    const int centerY0 = height / 4;
    const int centerY1 = height - centerY0;
    for (int y = 0; y < height; ++y) {
        const auto* row =
            reinterpret_cast<const std::uint16_t*>(base + y * stride);
        for (int x = 0; x < width; ++x) {
            const std::uint16_t value = row[x];
            ++quality.total_pixels;
            if (value == 0) {
                ++quality.zero_pixels;
            } else {
                ++quality.valid_pixels;
                quality.min_depth = std::min(quality.min_depth, value);
                quality.max_depth = std::max(quality.max_depth, value);
            }
            if (x >= centerX0 && x < centerX1 && y >= centerY0 &&
                y < centerY1) {
                ++quality.center_total_pixels;
                if (value == 0) {
                    ++quality.center_zero_pixels;
                } else {
                    ++quality.center_valid_pixels;
                    quality.center_min_depth =
                        std::min(quality.center_min_depth, value);
                    quality.center_max_depth =
                        std::max(quality.center_max_depth, value);
                    ++quality.center_histogram[value];
                }
            }
        }
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

StreamAudit auditStream(
    const std::string& inputPath,
    const std::string& streamName) {
    StreamAudit audit{streamName};
    const openni::SensorType sensorType = sensorTypeForName(streamName);
    openni::Device device;
    openni::Status status = device.open(inputPath.c_str());
    if (status != openni::STATUS_OK) {
        audit.errors.push_back(
            std::string("device_open_failed: ") + openni::OpenNI::getExtendedError());
        return audit;
    }
    audit.exists = device.hasSensor(sensorType);
    if (!audit.exists) {
        device.close();
        return audit;
    }

    openni::VideoStream stream;
    status = stream.create(device, sensorType);
    if (status != openni::STATUS_OK) {
        audit.errors.push_back(
            std::string("stream_create_failed: ") +
            openni::OpenNI::getExtendedError());
        device.close();
        return audit;
    }
    audit.create_success = true;
    const openni::VideoMode mode = stream.getVideoMode();
    audit.width = mode.getResolutionX();
    audit.height = mode.getResolutionY();
    audit.nominal_fps = mode.getFps();
    audit.pixel_format = pixelFormatName(mode.getPixelFormat());

    openni::PlaybackControl* playback = device.getPlaybackControl();
    if (playback == nullptr) {
        audit.errors.push_back("playback_control_unavailable");
        stream.destroy();
        device.close();
        return audit;
    }
    playback->setRepeatEnabled(false);
    playback->setSpeed(-1.0f);
    audit.expected_frame_count = playback->getNumberOfFrames(stream);
    if (audit.expected_frame_count < 0) {
        audit.expected_frame_count = 0;
    }

    status = stream.start();
    if (status != openni::STATUS_OK) {
        audit.errors.push_back(
            std::string("stream_start_failed: ") +
            openni::OpenNI::getExtendedError());
        stream.destroy();
        device.close();
        return audit;
    }
    audit.start_success = true;

    std::int64_t previousTimestamp = -1;
    int previousFrameIndex = -1;
    const int hardFrameLimit = audit.expected_frame_count > 0
                                   ? audit.expected_frame_count
                                   : 10'000'000;
    for (int iteration = 0; iteration < hardFrameLimit; ++iteration) {
        openni::VideoFrameRef frame;
        status = stream.readFrame(&frame);
        if (status != openni::STATUS_OK || !frame.isValid()) {
            ++audit.decode_error_count;
            audit.errors.push_back(
                std::string("frame_read_failed_after_") +
                std::to_string(audit.actual_frame_count) + "_frames: " +
                openni::OpenNI::getExtendedError());
            break;
        }
        const std::int64_t timestamp =
            static_cast<std::int64_t>(frame.getTimestamp());
        const int frameIndex = frame.getFrameIndex();
        if (audit.actual_frame_count == 0) {
            audit.first_timestamp_us = timestamp;
            audit.first_frame_index = frameIndex;
        } else {
            const std::int64_t interval = timestamp - previousTimestamp;
            if (interval <= 0) {
                audit.timestamps_strictly_increasing = false;
                ++audit.non_increasing_timestamp_count;
            } else {
                audit.positive_intervals_us.push_back(interval);
                if (audit.nominal_fps > 0) {
                    const double nominalInterval =
                        1'000'000.0 / static_cast<double>(audit.nominal_fps);
                    if (static_cast<double>(interval) >
                            nominalInterval * 1.8 ||
                        static_cast<double>(interval) <
                            nominalInterval * 0.2) {
                        ++audit.abnormal_interval_count;
                    }
                    const int estimatedGap = static_cast<int>(
                        std::llround(
                            static_cast<double>(interval) / nominalInterval)) -
                        1;
                    audit.estimated_dropped_frames +=
                        std::max(0, estimatedGap);
                }
            }
            if (frameIndex != previousFrameIndex + 1) {
                audit.frame_indices_continuous = false;
                ++audit.frame_index_discontinuity_count;
            }
        }
        previousTimestamp = timestamp;
        previousFrameIndex = frameIndex;
        audit.last_timestamp_us = timestamp;
        audit.last_frame_index = frameIndex;
        ++audit.actual_frame_count;
        if (streamName == "depth") {
            updateDepthQuality(frame, audit.depth);
        }
        if (audit.expected_frame_count > 0 &&
            audit.actual_frame_count >= audit.expected_frame_count) {
            break;
        }
    }

    stream.stop();
    stream.destroy();
    device.close();

    audit.interval_p50_us =
        percentile(audit.positive_intervals_us, 50.0);
    audit.interval_p95_us =
        percentile(audit.positive_intervals_us, 95.0);
    if (audit.actual_frame_count > 1 &&
        audit.last_timestamp_us > audit.first_timestamp_us) {
        audit.actual_fps =
            static_cast<double>(audit.actual_frame_count - 1) * 1'000'000.0 /
            static_cast<double>(
                audit.last_timestamp_us - audit.first_timestamp_us);
    }
    audit.complete =
        audit.start_success && audit.decode_error_count == 0 &&
        audit.actual_frame_count > 0 &&
        (audit.expected_frame_count == 0 ||
         audit.actual_frame_count == audit.expected_frame_count);
    return audit;
}

void classify(FileAudit& audit) {
    const bool anyStream =
        audit.color.exists || audit.depth.exists || audit.ir.exists;
    const bool allPresentComplete =
        (!audit.color.exists || audit.color.complete) &&
        (!audit.depth.exists || audit.depth.complete) &&
        (!audit.ir.exists || audit.ir.complete);
    audit.complete_playback =
        audit.open_success && anyStream && allPresentComplete;
    if (!audit.complete_playback) {
        audit.classification = "E";
        audit.classification_description = "corrupt_or_unreadable";
        audit.qualified_for_rgbd = false;
    } else if (audit.color.exists && audit.depth.exists) {
        audit.classification = "A";
        audit.classification_description = "color_and_depth";
        audit.qualified_for_rgbd = true;
    } else if (audit.depth.exists && !audit.color.exists) {
        audit.classification = "B";
        audit.classification_description = "depth_without_color";
        audit.qualified_for_rgbd = false;
    } else if (audit.color.exists && !audit.depth.exists) {
        audit.classification = "C";
        audit.classification_description = "color_without_depth";
        audit.qualified_for_rgbd = false;
    } else {
        audit.classification = "D";
        audit.classification_description = "ir_only_or_incomplete_stream_set";
        audit.qualified_for_rgbd = false;
    }
}

std::string nullableInteger(std::int64_t value) {
    return value < 0 ? "null" : std::to_string(value);
}

double ratio(std::uint64_t numerator, std::uint64_t denominator) {
    return denominator == 0
               ? 0.0
               : static_cast<double>(numerator) /
                     static_cast<double>(denominator);
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

void writeDepthJson(std::ostream& output, const DepthQuality& depth) {
    const bool hasValidDepth = depth.valid_pixels > 0;
    const bool centerHasValidDepth = depth.center_valid_pixels > 0;
    output << "{\n";
    output << "        \"depth_scale_mm\": " << depth.depth_scale_mm << ",\n";
    output << "        \"total_pixel_count\": " << depth.total_pixels << ",\n";
    output << "        \"valid_pixel_count\": " << depth.valid_pixels << ",\n";
    output << "        \"zero_pixel_count\": " << depth.zero_pixels << ",\n";
    output << "        \"zero_value_ratio\": "
           << ratio(depth.zero_pixels, depth.total_pixels) << ",\n";
    output << "        \"invalid_pixel_ratio\": "
           << ratio(depth.zero_pixels, depth.total_pixels) << ",\n";
    output << "        \"min_depth_raw\": "
           << (hasValidDepth ? std::to_string(depth.min_depth) : "null")
           << ",\n";
    output << "        \"max_depth_raw\": "
           << (hasValidDepth ? std::to_string(depth.max_depth) : "null")
           << ",\n";
    output << "        \"all_depth_invalid\": "
           << (hasValidDepth ? "false" : "true") << ",\n";
    output << "        \"center_region\": {\n";
    output << "          \"definition\": \"central_50_percent_width_and_height\",\n";
    output << "          \"total_pixel_count\": " << depth.center_total_pixels
           << ",\n";
    output << "          \"valid_pixel_count\": " << depth.center_valid_pixels
           << ",\n";
    output << "          \"zero_value_ratio\": "
           << ratio(depth.center_zero_pixels, depth.center_total_pixels)
           << ",\n";
    output << "          \"min_depth_raw\": "
           << (centerHasValidDepth
                   ? std::to_string(depth.center_min_depth)
                   : "null")
           << ",\n";
    output << "          \"p05_depth_raw\": "
           << (centerHasValidDepth
                   ? std::to_string(histogramPercentile(
                         depth.center_histogram,
                         depth.center_valid_pixels,
                         5.0))
                   : "null")
           << ",\n";
    output << "          \"p50_depth_raw\": "
           << (centerHasValidDepth
                   ? std::to_string(histogramPercentile(
                         depth.center_histogram,
                         depth.center_valid_pixels,
                         50.0))
                   : "null")
           << ",\n";
    output << "          \"p95_depth_raw\": "
           << (centerHasValidDepth
                   ? std::to_string(histogramPercentile(
                         depth.center_histogram,
                         depth.center_valid_pixels,
                         95.0))
                   : "null")
           << ",\n";
    output << "          \"max_depth_raw\": "
           << (centerHasValidDepth
                   ? std::to_string(depth.center_max_depth)
                   : "null")
           << "\n";
    output << "        }\n";
    output << "      }";
}

void writeStreamJson(std::ostream& output, const StreamAudit& stream) {
    output << "{\n";
    output << "      \"exists\": " << (stream.exists ? "true" : "false")
           << ",\n";
    output << "      \"create_success\": "
           << (stream.create_success ? "true" : "false") << ",\n";
    output << "      \"start_success\": "
           << (stream.start_success ? "true" : "false") << ",\n";
    output << "      \"complete\": " << (stream.complete ? "true" : "false")
           << ",\n";
    output << "      \"width\": " << stream.width << ",\n";
    output << "      \"height\": " << stream.height << ",\n";
    output << "      \"pixel_format\": \"" << jsonEscape(stream.pixel_format)
           << "\",\n";
    output << "      \"nominal_fps\": " << stream.nominal_fps << ",\n";
    output << "      \"expected_frame_count\": "
           << stream.expected_frame_count << ",\n";
    output << "      \"actual_frame_count\": " << stream.actual_frame_count
           << ",\n";
    output << "      \"first_timestamp_us\": "
           << nullableInteger(stream.first_timestamp_us) << ",\n";
    output << "      \"last_timestamp_us\": "
           << nullableInteger(stream.last_timestamp_us) << ",\n";
    output << "      \"first_frame_index\": "
           << nullableInteger(stream.first_frame_index) << ",\n";
    output << "      \"last_frame_index\": "
           << nullableInteger(stream.last_frame_index) << ",\n";
    output << "      \"timestamps_strictly_increasing\": "
           << (stream.timestamps_strictly_increasing ? "true" : "false")
           << ",\n";
    output << "      \"frame_indices_continuous\": "
           << (stream.frame_indices_continuous ? "true" : "false") << ",\n";
    output << "      \"non_increasing_timestamp_count\": "
           << stream.non_increasing_timestamp_count << ",\n";
    output << "      \"frame_index_discontinuity_count\": "
           << stream.frame_index_discontinuity_count << ",\n";
    output << "      \"interval_p50_us\": " << stream.interval_p50_us
           << ",\n";
    output << "      \"interval_p95_us\": " << stream.interval_p95_us
           << ",\n";
    output << "      \"actual_fps\": " << stream.actual_fps << ",\n";
    output << "      \"estimated_dropped_frames\": "
           << stream.estimated_dropped_frames << ",\n";
    output << "      \"abnormal_interval_count\": "
           << stream.abnormal_interval_count << ",\n";
    output << "      \"decode_error_count\": " << stream.decode_error_count
           << ",\n";
    output << "      \"errors\": ";
    writeStringArray(output, stream.errors);
    if (stream.name == "depth") {
        output << ",\n      \"depth_quality\": ";
        writeDepthJson(output, stream.depth);
    }
    output << "\n    }";
}

std::string serializeReport(
    const FileAudit& audit,
    const openni::Version& version) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(6);
    output << "{\n";
    output << "  \"schema_version\": 1,\n";
    output << "  \"artifact_type\": \"oni_inventory\",\n";
    output << "  \"tool\": {\n";
    output << "    \"name\": \"oni-inspect\",\n";
    output << "    \"version\": \"" << kToolVersion << "\",\n";
    output << "    \"openni_version\": \"" << version.major << "."
           << version.minor << "." << version.maintenance << "."
           << version.build << "\",\n";
    output << "    \"offline_file_only\": true\n";
    output << "  },\n";
    output << "  \"file\": {\n";
    output << "    \"path\": \"" << jsonEscape(audit.input_path_utf8)
           << "\",\n";
    output << "    \"size_bytes\": " << audit.file_size_bytes << ",\n";
    output << "    \"open_success\": "
           << (audit.open_success ? "true" : "false") << ",\n";
    output << "    \"complete_playback\": "
           << (audit.complete_playback ? "true" : "false") << ",\n";
    output << "    \"duration_us\": " << audit.duration_us << ",\n";
    output << "    \"decode_error_count\": "
           << (audit.color.decode_error_count +
               audit.depth.decode_error_count + audit.ir.decode_error_count)
           << ",\n";
    output << "    \"errors\": ";
    writeStringArray(output, audit.errors);
    output << "\n  },\n";
    output << "  \"classification\": {\n";
    output << "    \"code\": \"" << audit.classification << "\",\n";
    output << "    \"description\": \""
           << audit.classification_description << "\",\n";
    output << "    \"qualified_for_rgbd\": "
           << (audit.qualified_for_rgbd ? "true" : "false") << "\n";
    output << "  },\n";
    output << "  \"streams\": {\n";
    output << "    \"color\": ";
    writeStreamJson(output, audit.color);
    output << ",\n    \"depth\": ";
    writeStreamJson(output, audit.depth);
    output << ",\n    \"ir\": ";
    writeStreamJson(output, audit.ir);
    output << "\n  }\n";
    output << "}\n";
    return output.str();
}

void writeReport(
    const std::wstring& outputPath,
    const FileAudit& audit,
    const openni::Version& version) {
    const std::string report = serializeReport(audit, version);
#ifdef _WIN32
    std::FILE* output = _wfopen(outputPath.c_str(), L"wb");
#else
    const std::string narrowPath = wideToUtf8(outputPath);
    std::FILE* output = std::fopen(narrowPath.c_str(), "wb");
#endif
    if (output == nullptr) {
        throw std::runtime_error(
            "cannot open JSON output: " + wideToUtf8(outputPath));
    }
    const std::size_t bytesWritten =
        std::fwrite(report.data(), 1, report.size(), output);
    const int closeStatus = std::fclose(output);
    if (bytesWritten != report.size() || closeStatus != 0) {
        throw std::runtime_error(
            "cannot completely write JSON output: " +
            wideToUtf8(outputPath));
    }
}

int inspect(
    const std::wstring& inputWide,
    const std::wstring& outputWide) {
    FileAudit audit;
    audit.input_path_utf8 = wideToUtf8(inputWide);
    std::string fileSizeError;
    if (!getFileSize(inputWide, audit.file_size_bytes, fileSizeError)) {
        audit.errors.push_back("file_size_failed: " + fileSizeError);
    }

    const openni::Status initialization = openni::OpenNI::initialize();
    const openni::Version version = openni::OpenNI::getVersion();
    if (initialization != openni::STATUS_OK) {
        audit.errors.push_back(
            std::string("openni_initialize_failed: ") +
            openni::OpenNI::getExtendedError());
        writeReport(outputWide, audit, version);
        return 3;
    }

    std::wstring stagedInputPath;
    std::string openNiPathError;
    const std::string openNiPath =
        wideToOpenNiPath(inputWide, stagedInputPath, openNiPathError);
    if (openNiPath.empty()) {
        audit.errors.push_back(
            "openni_path_prepare_failed: " + openNiPathError);
        classify(audit);
        writeReport(outputWide, audit, version);
        openni::OpenNI::shutdown();
        return 3;
    }
    {
        openni::Device device;
        const openni::Status openStatus = device.open(openNiPath.c_str());
        audit.open_success = openStatus == openni::STATUS_OK;
        if (!audit.open_success) {
            audit.errors.push_back(
                std::string("device_open_failed: ") +
                openni::OpenNI::getExtendedError());
        }
        device.close();
    }

    if (audit.open_success) {
        audit.color = auditStream(openNiPath, "color");
        audit.depth = auditStream(openNiPath, "depth");
        audit.ir = auditStream(openNiPath, "ir");
        audit.duration_us = std::max({
            audit.color.last_timestamp_us - audit.color.first_timestamp_us,
            audit.depth.last_timestamp_us - audit.depth.first_timestamp_us,
            audit.ir.last_timestamp_us - audit.ir.first_timestamp_us,
            static_cast<std::int64_t>(0),
        });
    }
    classify(audit);
    writeReport(outputWide, audit, version);
    openni::OpenNI::shutdown();
#ifdef _WIN32
    if (!stagedInputPath.empty()) {
        if (!removeStagedHardLink(stagedInputPath, inputWide)) {
            std::cerr
                << "warning: could not remove temporary ONI hard link\n";
        }
    }
#endif
    return audit.classification == "E" ? 3 : 0;
}

void printUsage() {
    std::wcerr
        << L"Usage: oni-inspect.exe input.oni --output oni_inventory.json\n";
}

}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
    if (argc != 4 || std::wstring(argv[2]) != L"--output") {
        printUsage();
        return 2;
    }
    try {
        return inspect(argv[1], argv[3]);
    } catch (const std::exception& error) {
        std::cerr << "oni-inspect fatal error: " << error.what() << "\n";
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
        return inspect(
            std::wstring(argv[1], argv[1] + std::strlen(argv[1])),
            std::wstring(argv[3], argv[3] + std::strlen(argv[3])));
    } catch (const std::exception& error) {
        std::cerr << "oni-inspect fatal error: " << error.what() << "\n";
        return 4;
    }
}
#endif
