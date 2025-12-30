@echo off
REM AgriSense NPU Training - One-Click Execution
REM Intel Core Ultra 9 275HX Optimization

echo.
echo ╔════════════════════════════════════════════════════════════════════╗
echo ║  AgriSense NPU Model Training                                      ║
echo ║  Intel Core Ultra 9 275HX with NPU                                ║
echo ╚════════════════════════════════════════════════════════════════════╝
echo.

REM Check if NPU environment exists
if not exist "venv_npu" (
    echo ❌ NPU environment not found!
    echo.
    echo 🔧 Setting up NPU environment... This may take 15-20 minutes.
    echo.
    powershell -ExecutionPolicy Bypass -File setup_npu_environment.ps1
    
    if errorlevel 1 (
        echo.
        echo ❌ Setup failed! Please check the error messages above.
        pause
        exit /b 1
    )
    
    echo.
    echo ✅ NPU environment setup complete!
    echo.
)

REM Run training workflow
echo 🚀 Starting NPU training workflow...
echo.
powershell -ExecutionPolicy Bypass -File train_npu_models.ps1

if errorlevel 1 (
    echo.
    echo ❌ Training failed! Check logs above.
    pause
    exit /b 1
)

echo.
echo ╔════════════════════════════════════════════════════════════════════╗
echo ║  ✅ NPU TRAINING COMPLETE!                                         ║
echo ╚════════════════════════════════════════════════════════════════════╝
echo.
echo 📁 Models saved to: agrisense_app\backend\models\
echo 📖 Documentation: NPU_OPTIMIZATION_GUIDE.md
echo.

pause
