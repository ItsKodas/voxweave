# Downloads the pinned release of uv (https://github.com/astral-sh/uv, MIT OR Apache-2.0) for 64-bit Windows and
# puts uv.exe into installer/staged/bin. The archive must have the SHA-256 committed below, or this script fails
# and nothing is staged. uv.exe is bundled into the installer; it is never committed to git.
#
# To move to a newer uv: change $Version, and set $Sha256 to the content of
#   https://github.com/astral-sh/uv/releases/download/<version>/uv-x86_64-pc-windows-msvc.zip.sha256
param(
    [string]$Destination = (Join-Path $PSScriptRoot "..\installer\staged\bin")
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Version = "0.13.0"
$Sha256 = "088962f9e7b7bd9ea740c04c650b2a21c8928c345bd99ac24350dc924dba656c"
$Asset = "uv-x86_64-pc-windows-msvc.zip"
$Url = "https://github.com/astral-sh/uv/releases/download/$Version/$Asset"

$Destination = [System.IO.Path]::GetFullPath($Destination)
$Exe = Join-Path $Destination "uv.exe"
$Stamp = Join-Path $Destination "uv.version"
if ((Test-Path $Exe) -and (Test-Path $Stamp) -and ((Get-Content $Stamp -Raw).Trim() -eq "$Version $Sha256")) {
    Write-Host "uv $Version is already staged at $Exe"
    exit 0
}

$Work = Join-Path ([System.IO.Path]::GetTempPath()) ("voxweave-uv-" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $Work | Out-Null
try {
    $Zip = Join-Path $Work $Asset
    Write-Host "Downloading $Url"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $Url -OutFile $Zip -UseBasicParsing
    $Actual = (Get-FileHash -Path $Zip -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Actual -ne $Sha256) {
        throw "SHA-256 mismatch for $Asset $Version. Expected $Sha256, got $Actual. Nothing was staged."
    }
    $Unpacked = Join-Path $Work "unpacked"
    Expand-Archive -Path $Zip -DestinationPath $Unpacked
    $Found = Get-ChildItem -Path $Unpacked -Recurse -Filter "uv.exe" | Select-Object -First 1
    if ($null -eq $Found) { throw "uv.exe is not in $Asset" }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Copy-Item -Path $Found.FullName -Destination $Exe -Force
    Set-Content -Path $Stamp -Value "$Version $Sha256" -Encoding ascii
    Write-Host "Staged uv $Version at $Exe (archive SHA-256 verified)"
}
finally {
    Remove-Item -Recurse -Force -Path $Work -ErrorAction SilentlyContinue
}
