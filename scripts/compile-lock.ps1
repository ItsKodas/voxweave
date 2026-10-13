# Regenerates installer/requirements.lock: the exact versions and hashes of everything the installed app's
# Python environment gets. Inputs: backend/pyproject.toml (runtime dependencies plus the "transcribe" extra)
# and installer/requirements.in (faster-whisper). No dev dependencies. The plain onnxruntime package is left
# out on purpose; see installer/requirements.in.
#
# -Constraints <file> keeps packages at the versions listed in that file (for example the output of
# "uv pip freeze" for an environment the test suite has passed in). Without it, the newest versions are taken.
param(
    [string]$Constraints = ""
)
$ErrorActionPreference = "Stop"
$Repo = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
& (Join-Path $PSScriptRoot "fetch-uv.ps1")
if ($LASTEXITCODE -ne 0) { throw "uv could not be fetched" }
$Uv = Join-Path $Repo "installer\staged\bin\uv.exe"

$UvArgs = @(
    "pip", "compile", "backend/pyproject.toml", "installer/requirements.in",
    "--extra", "transcribe",
    "--no-emit-package", "onnxruntime",
    "--python-version", "3.12",
    "--python-platform", "x86_64-pc-windows-msvc",
    "--generate-hashes", "--no-annotate", "--quiet",
    "--custom-compile-command", "scripts/compile-lock.ps1",
    "-o", "installer/requirements.lock"
)
if ($Constraints -ne "") { $UvArgs += @("-c", [System.IO.Path]::GetFullPath($Constraints)) }

Push-Location $Repo
try {
    & $Uv @UvArgs
    if ($LASTEXITCODE -ne 0) { throw "uv pip compile failed" }
}
finally {
    Pop-Location
}
if (Select-String -Path (Join-Path $Repo "installer\requirements.lock") -Pattern "^onnxruntime==" -Quiet) {
    throw "The lock contains the plain onnxruntime package. It must never be installed."
}
Write-Host "Wrote installer/requirements.lock"
