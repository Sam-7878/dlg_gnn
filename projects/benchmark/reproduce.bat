@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0reproduce.ps1" %*
exit /b %ERRORLEVEL%
