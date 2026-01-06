"""
ML Pipeline Test Suite
Tests all components: Models, RAG, API integration
"""

import sys
from pathlib import Path
import numpy as np
import json

# Add backend to path
backend_path = Path(__file__).parent / "AGRISENSEFULL-STACK" / "AgriSense" / "agrisense_app" / "backend"
sys.path.insert(0, str(backend_path))

def test_model_loading():
    """Test if all models load successfully"""
    print("\n" + "="*60)
    print("🧪 TEST 1: Model Loading")
    print("="*60)
    
    try:
        from ml.inference import ModelInference
        
        engine = ModelInference()
        info = engine.get_model_info()
        
        print(f"✅ Models loaded: {len(info['models_loaded'])}")
        print(f"   Models: {', '.join(info['models_loaded'])}")
        print(f"   Status: {info['status']}")
        
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def test_rag_pipeline():
    """Test RAG pipeline initialization"""
    print("\n" + "="*60)
    print("🧪 TEST 2: RAG Pipeline")
    print("="*60)
    
    try:
        from ml.rag_pipeline import initialize_rag_pipeline
        
        pipeline = initialize_rag_pipeline()
        print("✅ RAG pipeline initialized")
        
        # Test intent classification
        intent, confidence = pipeline.intent_classifier.classify("What crops for Kharif?")
        print(f"✅ Intent classification: {intent} ({confidence:.2%} confidence)")
        
        # Test retrieval
        crops = pipeline.retriever.search_by_criteria(season="Kharif")
        print(f"✅ Crop retrieval: Found {len(crops)} Kharif crops")
        
        # Test full query
        response = pipeline.process_query("What crops should I grow in Kharif?", {'season': 'Kharif'})
        print(f"✅ Full RAG query: {response['intent']} - {response['confidence']:.2%}")
        print(f"   Response: {response['response_text'][:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ RAG pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_predictions():
    """Test individual model predictions"""
    print("\n" + "="*60)
    print("🧪 TEST 3: Model Predictions")
    print("="*60)
    
    try:
        from ml.inference import ModelInference
        
        engine = ModelInference()
        
        # Test crop type prediction
        features = np.random.randn(26)
        crop_type, probs = engine.predict_crop_type(features)
        print(f"✅ Crop Type: {crop_type}")
        print(f"   Top probability: {max(probs.values()):.2%}")
        
        # Test growth duration
        features = np.random.randn(23)
        days, metrics = engine.predict_growth_duration(features)
        print(f"✅ Growth Duration: {days:.0f} days")
        print(f"   Typical range: {metrics['typical_range']} days")
        
        # Test water requirement
        features = np.random.randn(19)
        water, metrics = engine.predict_water_requirement(features)
        print(f"✅ Water Requirement: {water:.2f} mm/day")
        print(f"   Typical range: {metrics['typical_range']} mm/day")
        
        # Test season prediction
        features = np.random.randn(20)
        season, probs = engine.predict_season(features)
        print(f"✅ Season: {season}")
        print(f"   Top probability: {max(probs.values()):.2%}")
        
        return True
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_endpoints():
    """Test FastAPI endpoints"""
    print("\n" + "="*60)
    print("🧪 TEST 4: API Endpoints")
    print("="*60)
    
    try:
        import asyncio
        from fastapi.testclient import TestClient
        from api.routes.ml_predictions import router
        from fastapi import FastAPI
        
        # Create test app
        app = FastAPI()
        app.include_router(router)
        
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        print(f"✅ Health check: {data['status']}")
        
        # Test models info
        response = client.get("/models/info")
        assert response.status_code == 200
        data = response.json()
        print(f"✅ Models info: {len(data['models'])} models loaded")
        
        # Test RAG query
        response = client.post("/rag/query", json={
            "query": "What crops for Kharif?",
            "season": "Kharif"
        })
        assert response.status_code == 200
        data = response.json()
        print(f"✅ RAG query endpoint: {data['intent']} intent")
        
        # Test crop search
        response = client.get("/crops/search?season=Kharif")
        assert response.status_code == 200
        data = response.json()
        print(f"✅ Crop search endpoint: Found {data['total']} crops")
        
        return True
    except Exception as e:
        print(f"❌ API endpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_files():
    """Test if all data files exist"""
    print("\n" + "="*60)
    print("🧪 TEST 5: Data Files")
    print("="*60)
    
    try:
        from pathlib import Path
        
        data_dir = Path(__file__).parent / "AGRISENSEFULL-STACK" / "AgriSense" / "agrisense_app" / "backend" / "data"
        
        # Check raw data
        raw_file = data_dir / "raw" / "india_crops_complete.csv"
        assert raw_file.exists()
        print(f"✅ Raw data: {raw_file.name}")
        
        # Check processed datasets
        datasets = ['crop_recommendation', 'crop_type_classification', 'growth_duration', 
                   'water_requirement', 'season_classification']
        for dataset in datasets:
            dataset_dir = data_dir / "processed" / dataset
            assert dataset_dir.exists()
            pkl_file = dataset_dir / f"{dataset}_complete.pkl"
            assert pkl_file.exists()
            print(f"✅ Dataset: {dataset}")
        
        # Check encoders
        encoders_dir = data_dir / "encoders"
        assert (encoders_dir / "label_encoders.json").exists()
        assert (encoders_dir / "scalers.pkl").exists()
        print(f"✅ Encoders: label_encoders.json, scalers.pkl")
        
        return True
    except Exception as e:
        print(f"❌ Data files test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("\n")
    print("🚀" * 30)
    print("   AGRISENSE ML PIPELINE TEST SUITE")
    print("🚀" * 30)
    
    tests = [
        ("Data Files", test_data_files),
        ("Model Loading", test_model_loading),
        ("RAG Pipeline", test_rag_pipeline),
        ("Predictions", test_predictions),
        ("API Endpoints", test_api_endpoints),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! ML pipeline is ready for integration.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
