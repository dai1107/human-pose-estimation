# ONI bridge third-party runtime

`oni-inspect.exe` is an offline file tool. Its OpenNI2 runtime is isolated in
this directory and is not copied into `src/`, the web application, or the
default Python environment.

- API headers: [OpenNI/OpenNI2](https://github.com/OpenNI/OpenNI2)
- Windows x64 Orbbec OpenNI2 2.3.0.65 redist:
  [yunswj/orbbec-Openni2](https://github.com/yunswj/orbbec-Openni2)
- License: Apache License 2.0, reproduced in `OPENNI2_LICENSE.txt`
- Included driver: `OniFile.dll` only
- Excluded driver: Orbbec live-camera driver

The MinGW-built executable also ships `libwinpthread-1.dll` from MinGW-w64.
Its runtime license is reproduced in `MINGW_W64_RUNTIME_LICENSE.txt`.
The upstream OpenNI DLLs depend on the separately installed Microsoft Visual
C++ 2013 x64 runtime (`MSVCR120.dll`).

Exact runtime file hashes are recorded in `runtime_manifest.json`.
