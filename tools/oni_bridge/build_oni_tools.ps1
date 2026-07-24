[CmdletBinding()]
param(
    [string]$Compiler = "D:\mingw64\bin\g++.exe",
    [string]$ToolRoot = ""
)

$Arguments = @(
    "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $PSScriptRoot "build_oni_inspect.ps1"),
    "-Compiler", $Compiler
)
if ($ToolRoot) {
    $Arguments += @("-ToolRoot", $ToolRoot)
}
& powershell @Arguments
exit $LASTEXITCODE
