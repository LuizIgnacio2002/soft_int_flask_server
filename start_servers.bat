@echo off
echo ============================================
echo Iniciando MLflow Server + Flask Serving
echo ============================================

echo [1/2] Iniciando MLflow Tracking Server (puerto 5001)...
start "MLflow Server" /min python -m mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri sqlite:///./mlflow.db --default-artifact-root ./mlruns
timeout /t 8 /nobreak >nul

echo [2/2] Iniciando Flask Serving Server (puerto 5000)...
start "Flask Serving" python serving/main.py

echo.
echo Listo!
echo - MLflow UI: http://localhost:5001
echo - Flask API: http://localhost:5000
echo - Signal endpoint: http://localhost:5000/signal
echo - Forecast endpoint: http://localhost:5000/consultar
echo.
echo Presiona Ctrl+C en las ventanas para detener.
pause
