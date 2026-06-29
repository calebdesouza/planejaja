@echo off
title NodeStor TUI
cd /d "%~dp0"
echo.
echo  Compilando nodestor-cli (debug)...
echo.
cargo build -p nodestor-cli
if %errorlevel% neq 0 (
    echo.
    echo  ERRO de compilacao. Verifique os logs acima.
    pause
    exit /b 1
)
echo.
echo  Iniciando TUI...
echo.
"D:\nodestor-target\x86_64-pc-windows-msvc\debug\nodestor.exe" tui
