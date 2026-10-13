# Downloads the pinned release of uv (https://github.com/astral-sh/uv, MIT OR Apache-2.0) for 64-bit Windows and
# puts uv.exe into installer/staged/bin. uv.exe is bundled into the installer; it is never committed to git.
#
# Two SHA-256 values are committed below: of the release archive and of the uv.exe inside it. The archive must
# match or nothing is staged. The uv.exe that ends up staged must match too, and that is checked every time,
# also for a file that is already there: a uv.exe left in the folder by anything else is replaced, never trusted.
#
# To move to a newer uv: change $Version, set $ArchiveSha256 to the content of
#   https://github.com/astral-sh/uv/releases/download/<version>/uv-x86_64-pc-windows-msvc.zip.sha256
# and set $ExeSha256 to the SHA-256 of the uv.exe in that archive (Get-FileHash after unpacking the verified
# archive; this script prints it when it does not match).
param(
    [string]$Destination = (Join-Path $PSScriptRoot "..\installer\staged\bin")
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Version = "0.13.0"
$ArchiveSha256 = "088962f9e7b7bd9ea740c04c650b2a21c8928c345bd99ac24350dc924dba656c"
$ExeSha256 = "2bfbe53d3cc95799e98fa08323b04844fc01d4f8c87dd5c00089e43f6554f162"
$Asset = "uv-x86_64-pc-windows-msvc.zip"
$Url = "https://github.com/astral-sh/uv/releases/download/$Version/$Asset"

function Get-Sha256([string]$Path) {
    (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

$Destination = [System.IO.Path]::GetFullPath($Destination)
$Exe = Join-Path $Destination "uv.exe"
if ((Test-Path $Exe) -and ((Get-Sha256 $Exe) -eq $ExeSha256)) {
    Write-Host "uv $Version is already staged at $Exe (SHA-256 verified)"
    exit 0
}

$Work = Join-Path ([System.IO.Path]::GetTempPath()) ("voxweave-uv-" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $Work | Out-Null
try {
    $Zip = Join-Path $Work $Asset
    Write-Host "Downloading $Url"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $Url -OutFile $Zip -UseBasicParsing
    $Actual = Get-Sha256 $Zip
    if ($Actual -ne $ArchiveSha256) {
        throw "SHA-256 mismatch for $Asset $Version. Expected $ArchiveSha256, got $Actual. Nothing was staged."
    }
    $Unpacked = Join-Path $Work "unpacked"
    Expand-Archive -Path $Zip -DestinationPath $Unpacked
    $Found = Get-ChildItem -Path $Unpacked -Recurse -Filter "uv.exe" | Select-Object -First 1
    if ($null -eq $Found) { throw "uv.exe is not in $Asset" }
    $Actual = Get-Sha256 $Found.FullName
    if ($Actual -ne $ExeSha256) {
        throw "SHA-256 mismatch for uv.exe from the verified $Asset $Version. Expected $ExeSha256, got $Actual. Nothing was staged."
    }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Copy-Item -Path $Found.FullName -Destination $Exe -Force
    if ((Get-Sha256 $Exe) -ne $ExeSha256) {
        Remove-Item -Force -Path $Exe
        throw "The staged uv.exe does not have the expected SHA-256 after copying. It was removed."
    }
    Write-Host "Staged uv $Version at $Exe (archive and uv.exe SHA-256 verified)"
}
finally {
    Remove-Item -Recurse -Force -Path $Work -ErrorAction SilentlyContinue
}
