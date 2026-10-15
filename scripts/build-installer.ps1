# Builds the Windows installer (NSIS, per-user, x64). Run by "npm run installer" in app/ and by the release
# workflow. Needs Node, Rust and an internet connection for the pinned uv; "npm ci" must have been run in app/.
#
#   1. checks that the version is the same everywhere (scripts/version.mjs)
#   2. stages what the installer bundles besides the app (scripts/stage-installer.ps1, which fetches uv)
#   3. runs "tauri build" with src-tauri/tauri.installer.conf.json, which adds the staged files as resources
#   4. checks that the installer script Tauri generated really installs them, and prints the result
#
# The bundle resources live in a separate Tauri config on purpose: Tauri copies resources next to development
# binaries too and fails when one is missing, so the main config (used by "npm run tauri dev") has none.
$ErrorActionPreference = "Stop"
# Started from PowerShell 7 (as on the GitHub runners), Windows PowerShell inherits its module path and then cannot
# load its own Get-FileHash, Invoke-WebRequest or Expand-Archive. Its own module folders are put back first; the
# scripts this one starts inherit them.
if ($PSVersionTable.PSEdition -eq "Desktop") {
    $env:PSModulePath = @(
        (Join-Path ([Environment]::GetFolderPath("MyDocuments")) "WindowsPowerShell\Modules"),
        (Join-Path $env:ProgramFiles "WindowsPowerShell\Modules"),
        (Join-Path $env:SystemRoot "system32\WindowsPowerShell\v1.0\Modules")
    ) -join ";"
}
$Repo = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$App = Join-Path $Repo "app"

function Invoke-Checked([string]$What, [scriptblock]$Command) {
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit code $LASTEXITCODE)" }
}

Invoke-Checked "The version check" { node (Join-Path $PSScriptRoot "version.mjs") check }
Invoke-Checked "Staging" { powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "stage-installer.ps1") }

# The app must start on a PC without the Visual C++ runtime installed, so that runtime is linked into it.
$env:RUSTFLAGS = "-C target-feature=+crt-static"
# Compiled into the app: only a build made here is the installed app (src-tauri/src/lib.rs, INSTALLER_BUILD).
# Files next to the program do not decide that, because Tauri leaves the resources of an earlier installer
# build in target/release for every later build to find.
$env:VOXWEAVE_INSTALLER = "1"
Push-Location $App
try {
    # --locked: exactly the crate versions in the committed Cargo.lock, or no build.
    Invoke-Checked "tauri build" { npm run tauri -- build --config src-tauri/tauri.installer.conf.json -- --locked }
}
finally {
    Pop-Location
}

$Nsis = Join-Path $App "src-tauri\target\release\nsis\x64\installer.nsi"
if (-not (Test-Path $Nsis)) { throw "The installer script was not generated: $Nsis" }
$Script = Get-Content $Nsis -Raw
foreach ($Needed in @("backend\aivoice\__main__.py", "backend\aivoice\engines\rvc.py", "backend\aivoice\prompts_en.txt",
                      "backend\pyproject.toml", "setup\bootstrap.py", "setup\requirements.lock", "bin\uv.exe")) {
    if (-not $Script.Contains($Needed)) { throw "The installer does not install $Needed" }
}
if ($Script.Contains("__pycache__")) { throw "The installer would install __pycache__ folders" }

$Installer = Get-ChildItem (Join-Path $App "src-tauri\target\release\bundle\nsis") -Filter "*-setup.exe" |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $Installer) { throw "No installer was produced" }
$Hash = (Get-FileHash -Path $Installer.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host ""
Write-Host "Installer: $($Installer.FullName)"
Write-Host ("Size:      {0:N1} MB" -f ($Installer.Length / 1MB))
Write-Host "SHA-256:   $Hash"
