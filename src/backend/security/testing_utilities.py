"""
Security Testing Utilities
Unit and integration test helpers for security features
"""
import json
import re
from typing import List, Tuple
from dataclasses import dataclass
import secrets


@dataclass
class SecurityTestCase:
    """Represents a security test case."""
    name: str
    description: str
    test_input: any
    expected_valid: bool
    expected_error: str = None


class SecurityTestSuite:
    """
    Collection of security test cases for validation.
    """

    @staticmethod
    def get_password_test_cases() -> List[SecurityTestCase]:
        """Get test cases for password validation."""
        return [
            SecurityTestCase(
                name="valid_strong_password",
                description="Strong password with all requirements",
                test_input="SecureP@ss123",
                expected_valid=True,
            ),
            SecurityTestCase(
                name="too_short_password",
                description="Password shorter than 12 characters",
                test_input="Weak@123",
                expected_valid=False,
                expected_error="must be at least 12 characters",
            ),
            SecurityTestCase(
                name="missing_uppercase",
                description="Password without uppercase letters",
                test_input="securepwd@1234",
                expected_valid=False,
                expected_error="must contain at least one uppercase letter",
            ),
            SecurityTestCase(
                name="missing_lowercase",
                description="Password without lowercase letters",
                test_input="SECUREPWD@1234",
                expected_valid=False,
                expected_error="must contain at least one lowercase letter",
            ),
            SecurityTestCase(
                name="missing_number",
                description="Password without numbers",
                test_input="SecurePassword@",
                expected_valid=False,
                expected_error="must contain at least one number",
            ),
            SecurityTestCase(
                name="missing_special_char",
                description="Password without special characters",
                test_input="SecurePassword123",
                expected_valid=False,
                expected_error="must contain at least one special character",
            ),
            SecurityTestCase(
                name="sequential_pattern",
                description="Password with sequential patterns (abc)",
                test_input="abcDefg@1234",
                expected_valid=False,
                expected_error="sequential pattern detected",
            ),
            SecurityTestCase(
                name="common_password",
                description="Common password (password123)",
                test_input="Password123@456",
                expected_valid=False,
                expected_error="common password detected",
            ),
        ]

    @staticmethod
    def get_injection_test_cases() -> List[SecurityTestCase]:
        """Get test cases for injection attack detection."""
        return [
            SecurityTestCase(
                name="sql_union_injection",
                description="SQL UNION-based injection",
                test_input="'; UNION SELECT * FROM users--",
                expected_valid=False,
            ),
            SecurityTestCase(
                name="sql_select_injection",
                description="SQL SELECT injection",
                test_input="1' OR '1'='1",
                expected_valid=False,
            ),
            SecurityTestCase(
                name="nosql_operator_injection",
                description="NoSQL operator ($ne) injection",
                test_input='{"$ne": null}',
                expected_valid=False,
            ),
            SecurityTestCase(
                name="nosql_regex_injection",
                description="NoSQL regex operator injection",
                test_input='{"$regex": ".*"}',
                expected_valid=False,
            ),
            SecurityTestCase(
                name="javascript_protocol",
                description="JavaScript protocol injection",
                test_input="javascript:alert('xss')",
                expected_valid=False,
            ),
            SecurityTestCase(
                name="xss_event_handler",
                description="XSS via event handler",
                test_input='"><img src=x onerror=alert("xss")>',
                expected_valid=False,
            ),
            SecurityTestCase(
                name="clean_alphanumeric",
                description="Clean alphanumeric string",
                test_input="device_sensor_001",
                expected_valid=True,
            ),
            SecurityTestCase(
                name="clean_email",
                description="Clean email format",
                test_input="farmer@agrisense.com",
                expected_valid=True,
            ),
        ]

    @staticmethod
    def get_xss_test_cases() -> List[SecurityTestCase]:
        """Get test cases for XSS vulnerability detection."""
        return [
            SecurityTestCase(
                name="script_tag_injection",
                description="Script tag injection",
                test_input="<script>alert('xss')</script>",
                expected_valid=False,
            ),
            SecurityTestCase(
                name="iframe_injection",
                description="IFrame injection",
                test_input='<iframe src="http://evil.com"></iframe>',
                expected_valid=False,
            ),
            SecurityTestCase(
                name="event_handler",
                description="Event handler injection",
                test_input='<img src=x onerror="alert(1)">',
                expected_valid=False,
            ),
            SecurityTestCase(
                name="data_uri",
                description="Data URI injection",
                test_input='<img src="data:text/html,<script>alert(1)</script>">',
                expected_valid=False,
            ),
            SecurityTestCase(
                name="svg_injection",
                description="SVG-based XSS",
                test_input='<svg onload="alert(\'xss\')"></svg>',
                expected_valid=False,
            ),
            SecurityTestCase(
                name="html_entity_encoding",
                description="HTML entity in text",
                test_input="&lt;script&gt;alert(1)&lt;/script&gt;",
                expected_valid=True,
            ),
            SecurityTestCase(
                name="plain_text",
                description="Plain text with no markup",
                test_input="This is a safe message for farmers",
                expected_valid=True,
            ),
        ]

    @staticmethod
    def get_file_upload_test_cases() -> List[SecurityTestCase]:
        """Get test cases for file upload validation."""
        return [
            SecurityTestCase(
                name="path_traversal_attack",
                description="Path traversal in filename",
                test_input="../../etc/passwd",
                expected_valid=False,
            ),
            SecurityTestCase(
                name="null_byte_injection",
                description="Null byte in filename",
                test_input="image.php\x00.jpg",
                expected_valid=False,
            ),
            SecurityTestCase(
                name="double_extension",
                description="Double extension file",
                test_input="image.php.jpg",
                expected_valid=False,
            ),
            SecurityTestCase(
                name="encoded_path_traversal",
                description="URL-encoded path traversal",
                test_input="%2e%2e%2fetc%2fpasswd",
                expected_valid=False,
            ),
            SecurityTestCase(
                name="valid_image_name",
                description="Valid image filename",
                test_input="sensor_reading_001.jpg",
                expected_valid=True,
            ),
            SecurityTestCase(
                name="valid_csv_name",
                description="Valid CSV filename",
                test_input="export_2024_01_15.csv",
                expected_valid=True,
            ),
        ]

    @staticmethod
    def get_sensor_data_test_cases() -> List[SecurityTestCase]:
        """Get test cases for sensor data validation."""
        return [
            SecurityTestCase(
                name="valid_temperature",
                description="Valid temperature reading",
                test_input={"sensor": "temperature", "value": 25.5},
                expected_valid=True,
            ),
            SecurityTestCase(
                name="temperature_too_high",
                description="Temperature exceeds maximum",
                test_input={"sensor": "temperature", "value": 150.0},
                expected_valid=False,
            ),
            SecurityTestCase(
                name="temperature_too_low",
                description="Temperature below minimum",
                test_input={"sensor": "temperature", "value": -100.0},
                expected_valid=False,
            ),
            SecurityTestCase(
                name="invalid_humidity",
                description="Humidity greater than 100%",
                test_input={"sensor": "humidity", "value": 150.0},
                expected_valid=False,
            ),
            SecurityTestCase(
                name="nan_value",
                description="NaN in sensor value",
                test_input={"sensor": "temperature", "value": float("nan")},
                expected_valid=False,
            ),
            SecurityTestCase(
                name="inf_value",
                description="Infinity in sensor value",
                test_input={"sensor": "humidity", "value": float("inf")},
                expected_valid=False,
            ),
        ]

    @staticmethod
    def get_url_validation_test_cases() -> List[SecurityTestCase]:
        """Get test cases for URL validation."""
        return [
            SecurityTestCase(
                name="valid_https_url",
                description="Valid HTTPS URL",
                test_input="https://api.agrisense.com/data",
                expected_valid=True,
            ),
            SecurityTestCase(
                name="javascript_protocol",
                description="JavaScript protocol URL",
                test_input="javascript:alert('xss')",
                expected_valid=False,
            ),
            SecurityTestCase(
                name="data_protocol",
                description="Data protocol URL",
                test_input="data:text/html,<script>alert(1)</script>",
                expected_valid=False,
            ),
            SecurityTestCase(
                name="file_protocol",
                description="File protocol URL",
                test_input="file:///etc/passwd",
                expected_valid=False,
            ),
            SecurityTestCase(
                name="vbscript_protocol",
                description="VBScript protocol URL",
                test_input="vbscript:msgbox('xss')",
                expected_valid=False,
            ),
            SecurityTestCase(
                name="valid_websocket_url",
                description="Valid WebSocket URL",
                test_input="wss://api.agrisense.com/ws",
                expected_valid=True,
            ),
        ]


class CryptoTestHelper:
    """Helper for cryptographic security testing."""

    @staticmethod
    def generate_test_token(length: int = 32) -> str:
        """Generate random token for testing."""
        return secrets.token_urlsafe(length)

    @staticmethod
    def generate_test_api_key(prefix: str = "test") -> str:
        """Generate test API key."""
        random_part = secrets.token_urlsafe(24)
        return f"{prefix}_{random_part}"

    @staticmethod
    def generate_test_device_id(prefix: str = "DEVICE") -> str:
        """Generate test device ID."""
        random_part = secrets.token_hex(4)
        return f"{prefix}_{random_part}".upper()


class InputValidationTestHelper:
    """Helper for input validation testing."""

    @staticmethod
    def test_email_validation() -> List[Tuple[str, bool]]:
        """Test email validation patterns."""
        return [
            ("user@example.com", True),
            ("john.doe@agrisense.co.uk", True),
            ("invalid.email@", False),
            ("@example.com", False),
            ("user@.com", False),
            ("user..name@example.com", True),  # Technically valid RFC
        ]

    @staticmethod
    def test_device_id_validation() -> List[Tuple[str, bool]]:
        """Test device ID validation patterns."""
        return [
            ("DEVICE_001", True),
            ("SENSOR-A-123", True),
            ("esp32_farm_01", True),
            ("INVALID DEVICE", False),
            ("device@123", False),
            ("device/123", False),
        ]

    @staticmethod
    def test_numeric_range_validation() -> List[Tuple[float, Tuple[float, float], bool]]:
        """Test numeric range validation."""
        return [
            (25.5, (0, 100), True),
            (-10, (-50, 150), True),
            (150, (0, 100), False),
            (-100, (0, 100), False),
            (0, (0, 100), True),
            (100, (0, 100), True),
        ]


class SecurityReportGenerator:
    """Generate security test reports."""

    @staticmethod
    def generate_test_report(
        test_name: str,
        total_tests: int,
        passed_tests: int,
        failed_tests: List[str],
    ) -> str:
        """Generate formatted test report."""
        report = f"""
╔════════════════════════════════════════════════════════════════════════╗
║                      SECURITY TEST REPORT                              ║
╠════════════════════════════════════════════════════════════════════════╣
║ Test Suite: {test_name:<60} ║
║ Total Tests: {total_tests:<57} ║
║ Passed: {passed_tests:<62} ║
║ Failed: {len(failed_tests):<62} ║
╠════════════════════════════════════════════════════════════════════════╣
"""

        if failed_tests:
            report += "║ Failed Tests:                                                              ║\n"
            for test in failed_tests:
                report += f"║ - {test:<70} ║\n"
        else:
            report += "║ ✓ All tests passed!                                                        ║\n"

        report += "╚════════════════════════════════════════════════════════════════════════╝\n"

        return report

    @staticmethod
    def generate_vulnerability_report(
        vulnerabilities: dict,
        severity_summary: dict,
    ) -> str:
        """Generate vulnerability report."""
        report = f"""
╔════════════════════════════════════════════════════════════════════════╗
║                   VULNERABILITY ASSESSMENT REPORT                      ║
╠════════════════════════════════════════════════════════════════════════╣
║                       Severity Summary                                  ║
╠════════════════════════════════════════════════════════════════════════╣
║ CRITICAL: {severity_summary.get('CRITICAL', 0):<53} ║
║ HIGH:     {severity_summary.get('HIGH', 0):<53} ║
║ MEDIUM:   {severity_summary.get('MEDIUM', 0):<53} ║
║ LOW:      {severity_summary.get('LOW', 0):<53} ║
╠════════════════════════════════════════════════════════════════════════╣
"""

        if vulnerabilities:
            report += "║ Affected Packages:                                                       ║\n"
            for package, vulns in vulnerabilities.items():
                report += f"║ - {package:<69} ║\n"
                for vuln in vulns:
                    report += f"║   • {vuln.get('cve_id', 'N/A'):<65} ║\n"
        else:
            report += "║ ✓ No vulnerabilities detected!                                             ║\n"

        report += "╚════════════════════════════════════════════════════════════════════════╝\n"

        return report
