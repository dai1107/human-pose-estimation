[CmdletBinding()]
param(
    [string]$Compiler = "D:\mingw64\bin\g++.exe",
    [string]$ToolRoot = ""
)

$ErrorActionPreference = "Stop"
if (-not $ToolRoot) {
    $ToolRoot = $PSScriptRoot
}
$ToolRoot = [System.IO.Path]::GetFullPath($ToolRoot)
$HeaderRoot = Join-Path $ToolRoot "vendor\OpenNIFork\Include"
$RedistRoot = Join-Path $ToolRoot "vendor\yunswj-orbbec\openi2 for orbbec\Redist"
$BuildRoot = Join-Path $ToolRoot "build"
$Gendef = "D:\mingw64\bin\gendef.exe"
$Dlltool = "D:\mingw64\bin\dlltool.exe"
$WinPthread = "D:\mingw64\bin\libwinpthread-1.dll"
$WinPthreadLicense = "D:\mingw64\licenses\mingw-w64\COPYING.MinGW-w64-runtime.txt"

foreach ($Required in @(
    $Compiler,
    $Gendef,
    $Dlltool,
    $WinPthread,
    $WinPthreadLicense,
    (Join-Path $HeaderRoot "OpenNI.h"),
    (Join-Path $RedistRoot "OpenNI2.dll"),
    (Join-Path $RedistRoot "OpenNI2\Drivers\OniFile.dll")
)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Missing ONI bridge build dependency: $Required"
    }
}

New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null
Push-Location $BuildRoot
try {
    & $Gendef (Join-Path $RedistRoot "OpenNI2.dll")
    if ($LASTEXITCODE -ne 0) {
        throw "gendef failed with exit code $LASTEXITCODE"
    }
    & $Dlltool -d "OpenNI2.def" -l "libOpenNI2.dll.a" -D "OpenNI2.dll"
    if ($LASTEXITCODE -ne 0) {
        throw "dlltool failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

$OutputExe = Join-Path $ToolRoot "oni-inspect.exe"
$CommonArguments = @(
    "-std=c++17",
    "-O2",
    "-Wall",
    "-Wextra",
    "-D_MSC_VER=1900",
    "-D_WIN32_WINNT=0x0A00",
    "-DWINVER=0x0A00",
    "-municode",
    "-static-libgcc",
    "-static-libstdc++",
    "-I$HeaderRoot"
)
$InspectorArguments = @(
    $CommonArguments
    (Join-Path $ToolRoot "oni_inspect.cpp"),
    "-L$BuildRoot",
    "-lOpenNI2",
    "-o",
    $OutputExe
)
& $Compiler @InspectorArguments
if ($LASTEXITCODE -ne 0) {
    throw "oni-inspect g++ failed with exit code $LASTEXITCODE"
}

$ExportExe = Join-Path $ToolRoot "oni-export.exe"
$ExporterArguments = @(
    $CommonArguments
    (Join-Path $ToolRoot "oni_export.cpp"),
    "-L$BuildRoot",
    "-lOpenNI2",
    "-o",
    $ExportExe
)
& $Compiler @ExporterArguments
if ($LASTEXITCODE -ne 0) {
    throw "oni-export g++ failed with exit code $LASTEXITCODE"
}

Copy-Item -LiteralPath (Join-Path $RedistRoot "OpenNI2.dll") -Destination $ToolRoot -Force
Copy-Item -LiteralPath $WinPthread -Destination $ToolRoot -Force
$DriverRoot = Join-Path $ToolRoot "OpenNI2\Drivers"
New-Item -ItemType Directory -Force -Path $DriverRoot | Out-Null
Copy-Item -LiteralPath (Join-Path $RedistRoot "OpenNI2\Drivers\OniFile.dll") -Destination $DriverRoot -Force
Copy-Item -LiteralPath (Join-Path $RedistRoot "OpenNI2\Drivers\OniFile.ini") -Destination $DriverRoot -Force
Copy-Item -LiteralPath (Join-Path $ToolRoot "vendor\pyOniExtractor\LICENSE") -Destination (Join-Path $ToolRoot "OPENNI2_LICENSE.txt") -Force
Copy-Item -LiteralPath $WinPthreadLicense -Destination (Join-Path $ToolRoot "MINGW_W64_RUNTIME_LICENSE.txt") -Force

$RuntimeFiles = @(
    $OutputExe,
    $ExportExe,
    (Join-Path $ToolRoot "OpenNI2.dll"),
    (Join-Path $ToolRoot "libwinpthread-1.dll"),
    (Join-Path $DriverRoot "OniFile.dll"),
    (Join-Path $DriverRoot "OniFile.ini"),
    (Join-Path $ToolRoot "OPENNI2_LICENSE.txt"),
    (Join-Path $ToolRoot "MINGW_W64_RUNTIME_LICENSE.txt")
)
$Manifest = [ordered]@{
    schema_version = 1
    artifact_type = "oni_bridge_runtime"
    generated_at = [DateTime]::UtcNow.ToString("o")
    architecture = "x86_64"
    openni_version = "2.3.0.65"
    offline_file_only = $true
    camera_driver_included = $false
    system_dependencies = @(
        "MSVCR120.dll (Microsoft Visual C++ 2013 x64 runtime)"
    )
    provenance = @{
        headers = "https://github.com/OpenNI/OpenNI2"
        runtime = "https://github.com/yunswj/orbbec-Openni2"
        runtime_upstream_note = "Orbbec OpenNI2 Redist"
        license = "Apache-2.0"
    }
    files = @(
        $RuntimeFiles | ForEach-Object {
            $RelativePath = $_.Substring($ToolRoot.Length).TrimStart("\")
            @{
                path = $RelativePath.Replace("\", "/")
                bytes = (Get-Item -LiteralPath $_).Length
                sha256 = (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLowerInvariant()
            }
        }
    )
}
$Manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $ToolRoot "runtime_manifest.json") -Encoding utf8

Write-Host "Built $OutputExe"
Write-Host "Built $ExportExe"
