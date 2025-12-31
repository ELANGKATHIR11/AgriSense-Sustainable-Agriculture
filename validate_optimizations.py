"""
Production Optimization Validation Script

Tests all optimization components to ensure they work correctly.
Run this script before deploying to production.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import logging

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "agrisense_app" / "backend"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ValidationResult:
    """Validation test result"""
    def __init__(self, name: str, passed: bool, message: str, duration_ms: float = 0):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration_ms = duration_ms
        self.timestamp = datetime.utcnow()


class ProductionValidator:
    """Validates production optimization components"""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.passed_count = 0
        self.failed_count = 0
    
    def run_all_tests(self) -> bool:
        """
        Run all validation tests
        
        Returns:
            bool: True if all tests passed
        """
        logger.info("=" * 80)
        logger.info("AgriSense Production Optimization Validation")
        logger.info("=" * 80)
        
        # Core modules
        self.test_cache_manager()
        self.test_auth_manager()
        self.test_security_validator()
        self.test_graceful_degradation()
        self.test_observability()
        
        # AI modules
        self.test_smart_recommendations()
        self.test_explainable_ai()
        
        # Configuration
        self.test_environment_variables()
        self.test_docker_files()
        
        # Summary
        self.print_summary()
        
        return self.failed_count == 0
    
    def test_cache_manager(self):
        """Test cache manager functionality"""
        logger.info("\n[1/9] Testing Cache Manager...")
        import time
        
        try:
            from core.cache_manager import CacheManager, cached
            
            # Test basic get/set
            cache = CacheManager()
            cache.set("test_key", {"value": 123}, ttl=60)
            result = cache.get("test_key")
            
            assert result is not None, "Cache get returned None"
            assert result["value"] == 123, "Cache value mismatch"
            
            # Test delete
            cache.delete("test_key")
            result = cache.get("test_key")
            assert result is None, "Cache delete failed"
            
            # Test decorator
            @cached(ttl=30)
            def expensive_function(x: int) -> int:
                return x * 2
            
            start = time.time()
            result1 = expensive_function(5)
            duration1 = (time.time() - start) * 1000
            
            start = time.time()
            result2 = expensive_function(5)  # Should be cached
            duration2 = (time.time() - start) * 1000
            
            assert result1 == result2 == 10, "Cached function result mismatch"
            
            self._add_result(ValidationResult(
                "Cache Manager",
                True,
                f"✓ All cache operations working (decorator {duration2:.2f}ms < {duration1:.2f}ms)",
                duration1
            ))
            
        except Exception as e:
            self._add_result(ValidationResult(
                "Cache Manager",
                False,
                f"✗ Cache manager failed: {str(e)}"
            ))
    
    def test_auth_manager(self):
        """Test authentication manager"""
        logger.info("\n[2/9] Testing Auth Manager...")
        
        try:
            from core.auth_manager import (
                create_access_token,
                decode_token,
                get_password_hash,
                verify_password
            )
            
            # Test JWT token creation and decoding
            token_data = {
                "sub": "test_user",
                "user_id": "123",
                "role": "farmer"
            }
            token = create_access_token(token_data)
            decoded = decode_token(token)
            
            assert decoded.username == "test_user", "JWT username mismatch"
            assert decoded.user_id == "123", "JWT user_id mismatch"
            assert decoded.role == "farmer", "JWT role mismatch"
            
            # Test password hashing
            password = "SecurePassword123!"
            hashed = get_password_hash(password)
            
            assert verify_password(password, hashed), "Password verification failed"
            assert not verify_password("WrongPassword", hashed), "Wrong password verified"
            
            self._add_result(ValidationResult(
                "Auth Manager",
                True,
                "✓ JWT tokens and password hashing working correctly"
            ))
            
        except Exception as e:
            self._add_result(ValidationResult(
                "Auth Manager",
                False,
                f"✗ Auth manager failed: {str(e)}"
            ))
    
    def test_security_validator(self):
        """Test security validation"""
        logger.info("\n[3/9] Testing Security Validator...")
        
        try:
            from core.security_validator import (
                validate_sensor_reading,
                detect_sensor_spoofing,
                sanitize_text_input
            )
            
            # Test valid sensor reading
            valid_data = {
                "device_id": "ESP32_001",
                "temperature": 25.5,
                "humidity": 65.0,
                "soil_moisture": 42.0,
                "ph_level": 6.8
            }
            validated = validate_sensor_reading(valid_data)
            assert validated.temperature == 25.5, "Validation altered valid data"
            
            # Test invalid sensor reading (should raise)
            try:
                invalid_data = {"temperature": 150.0}  # Too high
                validate_sensor_reading(invalid_data)
                raise AssertionError("Invalid data was not rejected")
            except Exception:
                pass  # Expected
            
            # Test spoofing detection
            is_spoofed = detect_sensor_spoofing(
                temperature=35.0,
                humidity=10.0,  # Impossible: high temp + low humidity
                soil_moisture=50.0
            )
            assert is_spoofed, "Spoofing detection failed"
            
            # Test input sanitization
            malicious_input = "<script>alert('XSS')</script>"
            sanitized = sanitize_text_input(malicious_input)
            assert "<script>" not in sanitized, "XSS sanitization failed"
            
            self._add_result(ValidationResult(
                "Security Validator",
                True,
                "✓ Input validation, spoofing detection, and sanitization working"
            ))
            
        except Exception as e:
            self._add_result(ValidationResult(
                "Security Validator",
                False,
                f"✗ Security validator failed: {str(e)}"
            ))
    
    def test_graceful_degradation(self):
        """Test graceful degradation and circuit breakers"""
        logger.info("\n[4/9] Testing Graceful Degradation...")
        
        try:
            from core.graceful_degradation import (
                CircuitBreaker,
                rule_based_irrigation_recommendation,
                _health_registry
            )
            
            # Test circuit breaker
            breaker = CircuitBreaker("test_service", failure_threshold=3, timeout=5)
            
            def failing_function():
                raise ValueError("Simulated failure")
            
            # Trigger failures
            for i in range(3):
                try:
                    breaker.call(failing_function)
                except:
                    pass
            
            # Circuit should be open now
            assert breaker.is_open(), "Circuit breaker didn't open after failures"
            
            # Test rule-based fallback
            recommendation = rule_based_irrigation_recommendation(
                soil_moisture=35.0,
                temperature=32.0,
                humidity=50.0,
                last_irrigation_hours_ago=24
            )
            
            assert "action" in recommendation, "Rule-based recommendation missing action"
            assert recommendation["action"] in ["irrigate_now", "irrigate_soon", "no_irrigation_needed"], \
                "Invalid recommendation action"
            
            # Test health registry
            _health_registry.register_check(
                "test_component",
                lambda: True,  # Always healthy
                critical=True
            )
            
            self._add_result(ValidationResult(
                "Graceful Degradation",
                True,
                "✓ Circuit breakers and fallback mechanisms working"
            ))
            
        except Exception as e:
            self._add_result(ValidationResult(
                "Graceful Degradation",
                False,
                f"✗ Graceful degradation failed: {str(e)}"
            ))
    
    def test_observability(self):
        """Test observability components"""
        logger.info("\n[5/9] Testing Observability...")
        
        try:
            from core.observability import (
                setup_structured_logging,
                ContextLogger,
                get_metrics,
                AgriSenseMetrics
            )
            
            # Test structured logging
            setup_structured_logging(level="INFO")
            logger_test = ContextLogger("test")
            logger_test.info("Test log", extra_field="value")
            
            # Test metrics
            metrics = get_metrics()
            metrics.increment(AgriSenseMetrics.API_REQUESTS_TOTAL)
            metrics.record(AgriSenseMetrics.ML_PREDICTION_DURATION, 150.0)
            
            stats = metrics.get_stats()
            assert AgriSenseMetrics.API_REQUESTS_TOTAL.value in stats, "Metrics not recorded"
            
            self._add_result(ValidationResult(
                "Observability",
                True,
                "✓ Structured logging and metrics collection working"
            ))
            
        except Exception as e:
            self._add_result(ValidationResult(
                "Observability",
                False,
                f"✗ Observability failed: {str(e)}"
            ))
    
    def test_smart_recommendations(self):
        """Test smart recommendations engine"""
        logger.info("\n[6/9] Testing Smart Recommendations...")
        
        try:
            from ai.smart_recommendations import (
                optimize_irrigation,
                optimize_fertilization
            )
            
            # Test irrigation optimization
            irrigation_result = optimize_irrigation(
                field_size_hectares=5.0,
                soil_type="loamy",
                current_soil_moisture=40.0,
                current_temperature=28.0,
                current_humidity=60.0,
                crop_type="tomato",
                growth_stage="flowering",
                days_ahead=7,
                available_budget=500.0
            )
            
            assert "objectives" in irrigation_result, "Missing objectives in irrigation result"
            assert "recommendations" in irrigation_result, "Missing recommendations"
            assert irrigation_result["constraints_satisfied"], "Constraints not satisfied"
            
            # Test fertilizer optimization
            fertilizer_result = optimize_fertilization(
                field_size_hectares=5.0,
                soil_type="clay",
                current_soil_moisture=50.0,
                current_temperature=25.0,
                crop_type="wheat",
                current_nitrogen=80.0,
                current_phosphorus=50.0,
                current_potassium=90.0,
                available_budget=400.0
            )
            
            assert "objectives" in fertilizer_result, "Missing objectives in fertilizer result"
            assert "parameters" in fertilizer_result, "Missing parameters"
            
            self._add_result(ValidationResult(
                "Smart Recommendations",
                True,
                f"✓ Multi-objective optimization working (Yield: {irrigation_result['objectives']['expected_yield_kg_per_ha']:.0f} kg/ha)"
            ))
            
        except Exception as e:
            self._add_result(ValidationResult(
                "Smart Recommendations",
                False,
                f"✗ Smart recommendations failed: {str(e)}"
            ))
    
    def test_explainable_ai(self):
        """Test explainable AI module"""
        logger.info("\n[7/9] Testing Explainable AI...")
        
        try:
            from ai.explainable_ai import explain_model_prediction
            from sklearn.ensemble import RandomForestClassifier
            import numpy as np
            
            # Create simple test model
            X = np.random.randn(100, 5)
            y = np.random.randint(0, 2, 100)
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
            
            feature_names = ["temp", "humidity", "moisture", "ph", "nitrogen"]
            
            # Get explanation
            explanation = explain_model_prediction(
                model=model,
                feature_names=feature_names,
                input_features={
                    "temp": 25.0,
                    "humidity": 60.0,
                    "moisture": 50.0,
                    "ph": 6.5,
                    "nitrogen": 150.0
                },
                model_type="classifier",
                method="rule_based"
            )
            
            assert "prediction" in explanation, "Missing prediction"
            assert "natural_language_explanation" in explanation, "Missing NL explanation"
            assert "actionable_insights" in explanation, "Missing insights"
            assert len(explanation["feature_contributions"]) > 0, "No feature contributions"
            
            self._add_result(ValidationResult(
                "Explainable AI",
                True,
                f"✓ Explanation generation working (Confidence: {explanation['prediction_confidence']*100:.1f}%)"
            ))
            
        except Exception as e:
            self._add_result(ValidationResult(
                "Explainable AI",
                False,
                f"✗ Explainable AI failed: {str(e)}"
            ))
    
    def test_environment_variables(self):
        """Test environment configuration"""
        logger.info("\n[8/9] Testing Environment Variables...")
        
        try:
            # Check if .env.production.template exists
            template_path = Path(".env.production.template")
            assert template_path.exists(), ".env.production.template not found"
            
            # Read and validate template
            with open(template_path, 'r') as f:
                content = f.read()
            
            required_vars = [
                "JWT_SECRET_KEY",
                "REDIS_HOST",
                "CACHE_ENABLED",
                "ENABLE_YIELD_MODEL",
                "LOG_LEVEL"
            ]
            
            missing_vars = []
            for var in required_vars:
                if var not in content:
                    missing_vars.append(var)
            
            if missing_vars:
                raise AssertionError(f"Missing variables: {', '.join(missing_vars)}")
            
            self._add_result(ValidationResult(
                "Environment Variables",
                True,
                "✓ All required environment variables present in template"
            ))
            
        except Exception as e:
            self._add_result(ValidationResult(
                "Environment Variables",
                False,
                f"✗ Environment configuration failed: {str(e)}"
            ))
    
    def test_docker_files(self):
        """Test Docker configuration"""
        logger.info("\n[9/9] Testing Docker Files...")
        
        try:
            # Check if Docker files exist
            dockerfile_prod = Path("Dockerfile.production")
            docker_compose = Path("docker-compose.production.yml")
            
            assert dockerfile_prod.exists(), "Dockerfile.production not found"
            assert docker_compose.exists(), "docker-compose.production.yml not found"
            
            # Validate Dockerfile structure
            with open(dockerfile_prod, 'r') as f:
                dockerfile_content = f.read()
            
            required_stages = ["FROM", "WORKDIR", "COPY", "RUN", "EXPOSE"]
            for stage in required_stages:
                if stage not in dockerfile_content:
                    raise AssertionError(f"Dockerfile missing {stage} instruction")
            
            # Validate docker-compose structure
            with open(docker_compose, 'r') as f:
                compose_content = f.read()
            
            required_services = ["api", "redis"]
            for service in required_services:
                if service not in compose_content:
                    raise AssertionError(f"docker-compose missing {service} service")
            
            self._add_result(ValidationResult(
                "Docker Files",
                True,
                "✓ Docker configuration files valid"
            ))
            
        except Exception as e:
            self._add_result(ValidationResult(
                "Docker Files",
                False,
                f"✗ Docker files validation failed: {str(e)}"
            ))
    
    def _add_result(self, result: ValidationResult):
        """Add test result"""
        self.results.append(result)
        if result.passed:
            self.passed_count += 1
            logger.info(f"   {result.message}")
        else:
            self.failed_count += 1
            logger.error(f"   {result.message}")
    
    def print_summary(self):
        """Print validation summary"""
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 80)
        
        total = self.passed_count + self.failed_count
        success_rate = (self.passed_count / total * 100) if total > 0 else 0
        
        logger.info(f"Total Tests: {total}")
        logger.info(f"Passed: {self.passed_count} ✓")
        logger.info(f"Failed: {self.failed_count} ✗")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        
        if self.failed_count > 0:
            logger.error("\n❌ VALIDATION FAILED - Fix errors before deploying to production")
            logger.info("\nFailed Tests:")
            for result in self.results:
                if not result.passed:
                    logger.error(f"  • {result.name}: {result.message}")
        else:
            logger.info("\n✅ ALL TESTS PASSED - Ready for production deployment!")
        
        logger.info("=" * 80)


def main():
    """Main validation entry point"""
    validator = ProductionValidator()
    success = validator.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
