# Collects everything the installer bundles besides the app itself into installer/staged (not in git):
#
#   staged/backend/aivoice/**      the voice engine's source (no __pycache__)
#   staged/backend/pyproject.toml
#   staged/setup/bootstrap.py      the first-run setup
#   staged/setup/requirements.lock the pinned packages it installs
#   staged/bin/uv.exe              the pinned uv, downloaded and checked by fetch-uv.ps1
#   staged/THIRD-PARTY-NOTICES.txt
#
# app/src-tauri/tauri.installer.conf.json maps these folders into the installer. Run by "npm run installer".
$ErrorActionPreference = "Stop"
$Repo = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$Staged = Join-Path $Repo "installer\staged"

& (Join-Path $PSScriptRoot "fetch-uv.ps1") -Destination (Join-Path $Staged "bin")
if ($LASTEXITCODE -ne 0) { throw "uv could not be fetched" }

foreach ($Name in @("backend", "setup")) {
    $Folder = Join-Path $Staged $Name
    if (Test-Path $Folder) { Remove-Item -Recurse -Force -Path $Folder }
    New-Item -ItemType Directory -Force -Path $Folder | Out-Null
}

$Source = Join-Path $Repo "backend\aivoice"
$Count = 0
Get-ChildItem -Path $Source -Recurse -File |
    Where-Object { $_.FullName -notmatch "\\__pycache__\\" -and $_.Extension -ne ".pyc" } |
    ForEach-Object {
        $Relative = $_.FullName.Substring($Source.Length + 1)
        $Target = Join-Path $Staged (Join-Path "backend\aivoice" $Relative)
        New-Item -ItemType Directory -Force -Path (Split-Path $Target -Parent) | Out-Null
        Copy-Item -Path $_.FullName -Destination $Target
        $Count++
    }
Copy-Item -Path (Join-Path $Repo "backend\pyproject.toml") -Destination (Join-Path $Staged "backend\pyproject.toml")
Copy-Item -Path (Join-Path $Repo "installer\bootstrap.py") -Destination (Join-Path $Staged "setup\bootstrap.py")
Copy-Item -Path (Join-Path $Repo "installer\requirements.lock") -Destination (Join-Path $Staged "setup\requirements.lock")
Copy-Item -Path (Join-Path $Repo "installer\THIRD-PARTY-NOTICES.txt") -Destination (Join-Path $Staged "THIRD-PARTY-NOTICES.txt")

foreach ($Needed in @("backend\aivoice\__main__.py", "backend\aivoice\prompts_en.txt", "backend\aivoice\engines\rvc.py",
                      "backend\pyproject.toml", "setup\bootstrap.py", "setup\requirements.lock", "bin\uv.exe")) {
    if (-not (Test-Path (Join-Path $Staged $Needed))) { throw "Staging is incomplete: $Needed is missing" }
}
Write-Host "Staged $Count backend files, the setup files and uv.exe in $Staged"
