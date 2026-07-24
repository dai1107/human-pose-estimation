[CmdletBinding()]
param(
    [string]$OutputDir = "reports/baseline",
    [int]$CameraIndex = 0,
    [switch]$SkipCamera,
    [switch]$SkipNode
)

$Arguments = @(
    "tools/run_baseline_regression.py",
    "--output-dir", $OutputDir,
    "--camera-index", $CameraIndex
)
if ($SkipCamera) {
    $Arguments += "--skip-camera"
}
if ($SkipNode) {
    $Arguments += "--skip-node"
}

& python @Arguments
exit $LASTEXITCODE
