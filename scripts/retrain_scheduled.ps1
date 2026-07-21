# BTC Forecast - Reentrenamiento programado (Task Scheduler)
# Usar con: powershell -ExecutionPolicy Bypass -File scripts\retrain_scheduled.ps1

$ErrorActionPreference = "Stop"
$projectDir = Split-Path $PSScriptRoot -Parent
$logDir = Join-Path $projectDir "logs"
$logFile = Join-Path $logDir "retrain_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

function Log {
    param($msg)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$timestamp] $msg" | Tee-Object -FilePath $logFile -Append
}

Log "=== Inicio reentrenamiento ==="

# Verificar MLflow
Log "Verificando MLflow server..."
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5001/health" -UseBasicParsing -TimeoutSec 5
    Log "MLflow server disponible"
} catch {
    Log "ERROR: MLflow server no disponible"
    exit 1
}

# Entrenar
Log "Entrenando modelos..."
Set-Location $projectDir
$trainOutput = python training\train.py 2>&1
$trainExit = $LASTEXITCODE
Log $trainOutput
if ($trainExit -ne 0) {
    Log "ERROR: Entrenamiento fallo (exit code: $trainExit)"
    exit 1
}
Log "Entrenamiento completado"

# Registrar
Log "Registrando modelo en Registry..."
$regOutput = python mlops\register_model.py 2>&1
$regExit = $LASTEXITCODE
Log $regOutput
if ($regExit -ne 0) {
    Log "ERROR: Registro fallo (exit code: $regExit)"
    exit 1
}
Log "Modelo registrado"

# Verificar nueva version
Log "Verificando version actual del modelo..."
$sigOutput = Invoke-WebRequest -Uri "http://localhost:5000/" -UseBasicParsing -TimeoutSec 5
$sigJson = $sigOutput.Content | ConvertFrom-Json
Log "Flask server respondiendo: modelo=$($sigJson.modelo) version=$($sigJson.version)"

Log "=== Reentrenamiento completado exitosamente ==="
