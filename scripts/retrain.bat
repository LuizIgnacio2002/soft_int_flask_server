@echo off
echo ============================================
echo BTC Forecast - Reentrenamiento Automatico
echo ============================================
echo.

cd /d "%~dp0.."

echo [1/4] Verificando MLflow server...
curl -s http://localhost:5001/health >nul 2>&1
if errorlevel 1 (
    echo [ERROR] MLflow server no esta corriendo en puerto 5001
    echo Inicialo con: python -m mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri sqlite:///./mlflow.db
    pause
    exit /b 1
)
echo [OK] MLflow server disponible

echo.
echo [2/4] Entrenando modelos...
python training\train.py
if errorlevel 1 (
    echo [ERROR] Entrenamiento fallo
    pause
    exit /b 1
)
echo [OK] Entrenamiento completado

echo.
echo [3/4] Registrando mejor modelo en MLflow Registry...
python mlops\register_model.py
if errorlevel 1 (
    echo [ERROR] Registro de modelo fallo
    pause
    exit /b 1
)
echo [OK] Modelo registrado

echo.
echo [4/4] El Flask server detectara automaticamente la nueva version
echo       No es necesario reiniciarlo (auto-reload cada 3 minutos)
echo.
echo ============================================
echo Reentrenamiento completado exitosamente
echo ============================================
echo.
pause
