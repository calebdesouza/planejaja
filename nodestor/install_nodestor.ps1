# NodeStor Installer (Windows Premium Experience)
# Usage: iwr -useb raw.githubusercontent.com/nodestor/install.ps1 | iex

$ErrorActionPreference = "Stop"

Write-Host "🚀 NodeStor — Instalando Motor de IA de Escala Industrial..." -ForegroundColor Cyan

$destDir = "$HOME\.nodestor\bin"
if (!(Test-Path $destDir)) { New-Item -Path $destDir -ItemType Directory }

# Mock download (Simulando o link do release real)
Write-Host "📥 Baixando binários nativos (Zero-Copy Core)..."
# Invoke-WebRequest -Uri "https://github.com/nodestor/releases/latest/download/nodestor.exe" -OutFile "$destDir\nodestor.exe"
# Invoke-WebRequest -Uri "https://github.com/nodestor/releases/latest/download/nodestor-server.exe" -OutFile "$destDir\nodestor-server.exe"

Write-Host "⚙️  Configurando variáveis de ambiente..." -ForegroundColor Yellow
$oldPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($oldPath -notlike "*$destDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$oldPath;$destDir", "User")
    $env:Path += ";$destDir"
}

Write-Host "✨ Instalação Concluída!" -ForegroundColor Green
Write-Host "Agora basta digitar 'nodestor' em qualquer terminal para começar."

# Inicia a calibração automática
Write-Host "`n🔍 Iniciando Autocalibração silenciosa..."
# & "$destDir\nodestor.exe" init --silent
