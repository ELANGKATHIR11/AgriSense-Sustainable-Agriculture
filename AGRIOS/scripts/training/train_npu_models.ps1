# ===================================================================
# AgriSense NPU Model Training - Complete Workflow
# Intel Core Ultra 9 275HX Optimization
# ===================================================================

param(
    [switch]$SkipBenchmark,
    [switch]$SkipConversion,
    [switch]$Quick
)

Write-Host "
╔════════════════════════════════════════════════════════════════════╗
║  AgriSense NPU-Optimized Model Training                           ║
║  Intel Core Ultra 9 275HX with NPU                                ║
║  Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')                             ║
╚════════════════════════════════════════════════════════════════════╝
" -ForegroundColor Cyan

# Check if NPU environment exists
if (-not (Test-Path "venv_npu")) {
    Write-Host "❌ NPU environment not found!" -ForegroundColor Red
    Write-Host "🔧 Please run: .\setup_npu_environment.ps1" -ForegroundColor Yellow
    exit 1
}

# Activate NPU environment
Write-Host "`n📦 Activating NPU environment..." -ForegroundColor Yellow
& .\venv_npu\Scripts\Activate.ps1

# Verify activation
$pythonPath = python -c "import sys; print(sys.executable)"
Write-Host "   Python: $pythonPath" -ForegroundColor Green

# Step 1: Check NPU availability
Write-Host "`n" + ("=" * 70) -ForegroundColor Cyan
Write-Host "STEP 1: NPU Device Detection" -ForegroundColor Cyan
Write-Host ("=" * 70) -ForegroundColor Cyan

python tools/npu/check_npu_devices.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n⚠️ Warning: NPU check failed, but continuing..." -ForegroundColor Yellow
}

# Step 2: Hardware Benchmark (optional)
if (-not $SkipBenchmark -and -not $Quick) {
    Write-Host "`n" + ("=" * 70) -ForegroundColor Cyan
    Write-Host "STEP 2: Hardware Benchmarking" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
    
    Write-Host "`n⏱️ This may take 5-10 minutes..." -ForegroundColor Yellow
    python tools/npu/benchmark_hardware.py
    
    if (Test-Path "npu_benchmark_results.json") {
        Write-Host "`n✅ Benchmark results saved: npu_benchmark_results.json" -ForegroundColor Green
    }
} else {
    Write-Host "`n⏭️ Skipping hardware benchmark..." -ForegroundColor Yellow
}

# Step 3: NPU-Optimized Training
Write-Host "`n" + ("=" * 70) -ForegroundColor Cyan
Write-Host "STEP 3: NPU-Optimized Model Training" -ForegroundColor Cyan
Write-Host ("=" * 70) -ForegroundColor Cyan

Write-Host "`n🧠 Training models with Intel acceleration..." -ForegroundColor Yellow
Write-Host "   - Random Forest (Intel oneDAL)" -ForegroundColor White
Write-Host "   - Gradient Boosting (Intel oneDAL)" -ForegroundColor White
Write-Host "   - Neural Network (IPEX + OpenVINO)" -ForegroundColor White

$trainingStart = Get-Date
python tools/npu/train_npu_optimized.py
$trainingEnd = Get-Date
$trainingDuration = ($trainingEnd - $trainingStart).TotalSeconds

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Training completed in $([math]::Round($trainingDuration, 2)) seconds!" -ForegroundColor Green
} else {
    Write-Host "`n❌ Training failed!" -ForegroundColor Red
    exit 1
}

# Step 4: Model Conversion (optional)
if (-not $SkipConversion -and -not $Quick) {
    Write-Host "`n" + ("=" * 70) -ForegroundColor Cyan
    Write-Host "STEP 4: OpenVINO Model Conversion" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
    
    Write-Host "`n🔄 Converting models to OpenVINO IR..." -ForegroundColor Yellow
    python tools/npu/convert_to_openvino.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✅ Model conversion completed!" -ForegroundColor Green
    } else {
        Write-Host "`n⚠️ Conversion failed, but trained models are available" -ForegroundColor Yellow
    }
} else {
    Write-Host "`n⏭️ Skipping OpenVINO conversion..." -ForegroundColor Yellow
}

# Step 5: Validate Models
Write-Host "`n" + ("=" * 70) -ForegroundColor Cyan
Write-Host "STEP 5: Model Validation" -ForegroundColor Cyan
Write-Host ("=" * 70) -ForegroundColor Cyan

Write-Host "`n🔍 Checking trained models..." -ForegroundColor Yellow

$modelsDir = "agrisense_app\backend\models"
$expectedModels = @(
    "crop_recommendation_rf_npu.joblib",
    "crop_recommendation_gb_npu.joblib",
    "crop_scaler.joblib",
    "crop_encoder.joblib"
)

$allFound = $true
foreach ($model in $expectedModels) {
    $path = Join-Path $modelsDir $model
    if (Test-Path $path) {
        $size = (Get-Item $path).Length / 1MB
        Write-Host "   ✅ $model ($([math]::Round($size, 2)) MB)" -ForegroundColor Green
    } else {
        Write-Host "   ❌ $model (not found)" -ForegroundColor Red
        $allFound = $false
    }
}

# Check OpenVINO models
$openvino_dir = Join-Path $modelsDir "openvino_npu"
if (Test-Path $openvino_dir) {
    $irModels = Get-ChildItem -Path $openvino_dir -Recurse -Filter "*.xml" | Measure-Object
    Write-Host "`n   🎯 OpenVINO IR models: $($irModels.Count) found" -ForegroundColor Cyan
}

# Step 6: Summary
Write-Host "`n" + ("=" * 70) -ForegroundColor Cyan
Write-Host "📊 TRAINING SUMMARY" -ForegroundColor Cyan
Write-Host ("=" * 70) -ForegroundColor Cyan

if ($allFound) {
    Write-Host "`n✅ All models trained successfully!" -ForegroundColor Green
} else {
    Write-Host "`n⚠️ Some models missing - check training logs" -ForegroundColor Yellow
}

# Read metrics if available
$metricsFile = Join-Path $modelsDir "npu_training_metrics.json"
if (Test-Path $metricsFile) {
    Write-Host "`n📈 Training Metrics:" -ForegroundColor Yellow
    $metrics = Get-Content $metricsFile | ConvertFrom-Json
    
    if ($metrics.training_times) {
        Write-Host "`n   ⏱️ Training Times:" -ForegroundColor White
        $metrics.training_times.PSObject.Properties | ForEach-Object {
            Write-Host "      $($_.Name): $([math]::Round($_.Value, 2))s" -ForegroundColor White
        }
    }
    
    if ($metrics.accuracies) {
        Write-Host "`n   🎯 Test Accuracies:" -ForegroundColor White
        $metrics.accuracies.PSObject.Properties | ForEach-Object {
            $percentage = [math]::Round($_.Value * 100, 2)
            Write-Host "      $($_.Name): $percentage%" -ForegroundColor White
        }
    }
}

# Performance Recommendations
Write-Host "`n" + ("=" * 70) -ForegroundColor Cyan
Write-Host "🚀 NEXT STEPS" -ForegroundColor Cyan
Write-Host ("=" * 70) -ForegroundColor Cyan

Write-Host "
1. 🔬 Test Models:
   python tests/test_model_accuracy.py

2. 📊 Compare Performance:
   python tools/npu/compare_performance.py

3. 🔧 Integrate with Backend:
   - Update model paths in backend configuration
   - Use OpenVINO IR models for NPU inference
   - Add performance monitoring

4. 🚀 Deploy:
   - Run validation tests
   - Update production environment
   - Monitor inference metrics

5. 📖 Documentation:
   - Read: NPU_OPTIMIZATION_GUIDE.md
   - Quick ref: NPU_QUICK_START.md
" -ForegroundColor White

Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host "✅ NPU TRAINING WORKFLOW COMPLETE!" -ForegroundColor Green
Write-Host ("=" * 70) -ForegroundColor Cyan

Write-Host "
💡 Performance Tips:
   - Use NPU for inference (10-50x faster)
   - Batch requests for higher throughput
   - Monitor with performance metrics
   - Retrain monthly with new data
" -ForegroundColor Cyan

Write-Host "`n📁 Models saved to: $modelsDir`n" -ForegroundColor White

# Exit with success
exit 0
