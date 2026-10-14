@echo off
rem Starts Voxweave from this folder: double-click this file.
rem Close the app with Quit in its tray menu; closing the window only hides it.
setlocal
cd /d "%~dp0app"

if not exist node_modules (
  echo Installing the app's packages the first time. This takes a minute.
  call npm install || goto :failed
)

rem A web server left over from an earlier run of this app can hold port 1420 and stop the app from starting.
rem Only a Node process running this folder's own vite is ended.
powershell -NoProfile -Command "$here = (Resolve-Path '.').Path; Get-CimInstance Win32_Process -Filter \"Name='node.exe'\" | Where-Object { $_.CommandLine -like ('*' + $here + '*vite*') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>&1

echo Starting Voxweave. Keep this window open while the app runs.
call npm run tauri dev
if errorlevel 1 goto :failed
exit /b 0

:failed
echo.
echo Voxweave could not start. The messages above say why.
pause
exit /b 1
