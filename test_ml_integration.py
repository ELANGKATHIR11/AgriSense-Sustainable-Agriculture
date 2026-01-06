#!/usr/bin/env python3
"""
Quick test of ML integration
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "src" / "backend"
sys.path.insert(0, str(backend_path))

print("=" * 70)
print("🧪 ML INTEGRATION TEST")
print("=" * 70)

# Test 1: Import ML modules
print("\n[1/5] Testing ML module imports...")
try:
    from ml.inference import get_inference_engine
    from ml.rag_pipeline import initialize_rag_pipeline
    print("✅ ML modules imported successfully")
except Exception as e:
    print(f"❌ ML module import failed: {e}")
    sys.exit(1)

# Test 2: Initialize Inference Engine
print("\n[2/5] Initializing ML Inference Engine...")
try:
    engine = get_inference_engine()
    print("✅ Inference Engine loaded")
    info = engine.get_model_info()
    print(f"   - Models loaded: {len(info.get('models', []))}")
    print(f"   - Metrics: {list(info.get('metrics', {}).keys())}")
except Exception as e:
    print(f"❌ Inference Engine failed: {e}")
    sys.exit(1)

# Test 3: Initialize RAG Pipeline
print("\n[3/5] Initializing RAG Pipeline...")
try:
    pipeline = initialize_rag_pipeline()
    print("✅ RAG Pipeline initialized")
except Exception as e:
    print(f"❌ RAG Pipeline failed: {e}")
    sys.exit(1)

# Test 4: Test RAG query
print("\n[4/5] Testing RAG query processing...")
try:
    result = pipeline.process_query(
        "What crops should I grow in Kharif?",
        {"season": "kharif"}
    )
    print("✅ RAG query processed successfully")
    print(f"   - Intent: {result.get('intent')}")
    print(f"   - Confidence: {result.get('confidence'):.2%}")
    print(f"   - Response preview: {result.get('response_text')[:80]}...")
except Exception as e:
    print(f"❌ RAG query failed: {e}")
    sys.exit(1)

# Test 5: Test predictions
print("\n[5/5] Testing ML predictions...")
try:
    from ml.inference import make_prediction
    import numpy as np
    
    # Test features for crop_type (26 features required)
    test_features = np.array([25.5, 60.0, 800.0, 6.5, 50, 15, 100, 75, 20, 5, 40, 30, 1, 0, 0, 1, 0, 0, 0, 0.5, 2.5, 1.2, 0.8, 0.3, 10, 5])
    
    result = make_prediction("crop_type", test_features)
    print("✅ ML prediction successful")
    print(f"   - Prediction: {result.get('crop_type')}")
    print(f"   - Probabilities: {list(result.get('probabilities', {}).values())[:3]}...")
except Exception as e:
    print(f"❌ Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED")
print("=" * 70)
print("\nNow test the API endpoints:")
print("  1. Start backend: cd src/backend && python -m uvicorn main:app --port 8004")
print("  2. Test health: curl http://localhost:8004/api/v1/ml/health")
print("  3. Test RAG: curl -X POST http://localhost:8004/api/v1/ml/rag/query \\")
print('       -H "Content-Type: application/json" \\')
print('       -d \'{"query": "What crops for Kharif?", "season": "kharif"}\'')
print("=" * 70)
