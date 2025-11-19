"""
Cleanup Sprint Verification Test Script
Tests Days 1-3 implementation: Error Handling, Validation, Logging & Config
"""
import os
import sys
import json
from pathlib import Path

# Try to import requests, handle gracefully if not available
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: 'requests' library not installed. API tests will be skipped.")
    print("Install with: pip install requests")

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


class CleanupVerifier:
    def __init__(self):
        self.base_url = "http://localhost:8000"  # FastAPI default port
        self.passed = 0
        self.failed = 0
        self.results = []
        self.backend_running = False

    def test(self, name, condition, error_msg=""):
        """Log test result"""
        if condition:
            print(f"{GREEN}✓ PASS{RESET}: {name}")
            self.passed += 1
        else:
            print(f"{RED}✗ FAIL{RESET}: {name}")
            if error_msg:
                print(f"  {RED}→ {error_msg}{RESET}")
            self.failed += 1
        self.results.append((name, condition))

    def section(self, title):
        """Print section header"""
        print(f"\n{YELLOW}{'='*60}{RESET}")
        print(f"{YELLOW}{title}{RESET}")
        print(f"{YELLOW}{'='*60}{RESET}\n")

    # TEST PART 1: LOGGING (Day 3)
    def test_logging(self):
        self.section("PART 1: TESTING LOGGING (Day 3)")

        # Check logs directory exists
        logs_dir = Path("backend/logs")
        self.test(
            "Logs directory exists",
            logs_dir.exists(),
            f"Expected directory: {logs_dir}"
        )

        # Check app.log file exists
        log_file = Path("backend/logs/app.log")
        self.test(
            "Logs file exists (app.log)",
            log_file.exists(),
            f"Expected file: {log_file}"
        )

        # Check logs have content
        if log_file.exists():
            with open(log_file, 'r') as f:
                content = f.read()
                self.test(
                    "Logs file has content",
                    len(content) > 0,
                    "Log file is empty"
                )
                self.test(
                    "Logs contain log levels (INFO/DEBUG/ERROR)",
                    any(level in content for level in ["INFO", "DEBUG", "ERROR", "WARNING"]),
                    "No log levels found in logs"
                )
                self.test(
                    "Logs contain timestamps",
                    "-" in content and ":" in content,  # Timestamp format
                    "No timestamps found in logs"
                )
                self.test(
                    "Logs contain function names",
                    "()" in content or " - " in content,
                    "No function names found in logs"
                )

        # Check other log files exist
        error_log = Path("backend/logs/errors.log")
        api_log = Path("backend/logs/api.log")
        db_log = Path("backend/logs/database.log")
        
        self.test(
            "Error log file exists",
            error_log.exists(),
            f"Expected file: {error_log}"
        )
        self.test(
            "API log file exists",
            api_log.exists(),
            f"Expected file: {api_log}"
        )
        self.test(
            "Database log file exists",
            db_log.exists(),
            f"Expected file: {db_log}"
        )

    # TEST PART 2: CONFIG (Day 3)
    def test_config(self):
        self.section("PART 2: TESTING CONFIG (Day 3)")

        # Check config file exists (actual location)
        config_file = Path("backend/core/config.py")
        self.test(
            "Config file exists (backend/core/config.py)",
            config_file.exists(),
            "backend/core/config.py not found"
        )

        if config_file.exists():
            with open(config_file, 'r') as f:
                config_content = f.read()

                self.test(
                    "Config has APPROVED_CITIES",
                    "APPROVED_CITIES" in config_content,
                    "APPROVED_CITIES list not defined"
                )
                self.test(
                    "Config has VALID_TRANSACTION_TYPES",
                    "VALID_TRANSACTION_TYPES" in config_content,
                    "VALID_TRANSACTION_TYPES list not defined"
                )
                self.test(
                    "Config has TOP_PRODUCTS_LIMIT",
                    "TOP_PRODUCTS_LIMIT" in config_content,
                    "TOP_PRODUCTS_LIMIT not defined"
                )
                self.test(
                    "Config has MAX_DATE_RANGE_DAYS",
                    "MAX_DATE_RANGE_DAYS" in config_content,
                    "MAX_DATE_RANGE_DAYS not defined"
                )
                self.test(
                    "Config has MOVER_GROWTH_THRESHOLD",
                    "MOVER_GROWTH_THRESHOLD" in config_content,
                    "MOVER_GROWTH_THRESHOLD not defined"
                )
                self.test(
                    "Config has Settings class",
                    "class Settings" in config_content,
                    "Settings class not found"
                )

        # Check logger file exists
        logger_file = Path("backend/utils/logger.py")
        self.test(
            "Logger utility file exists",
            logger_file.exists(),
            "backend/utils/logger.py not found"
        )

        if logger_file.exists():
            with open(logger_file, 'r') as f:
                logger_content = f.read()
                self.test(
                    "Logger has setup_logger function",
                    "def setup_logger" in logger_content,
                    "setup_logger function not found"
                )
                self.test(
                    "Logger has file handler",
                    "FileHandler" in logger_content,
                    "File handler not configured"
                )
                self.test(
                    "Logger has console handler",
                    "StreamHandler" in logger_content,
                    "Console handler not configured"
                )

    # TEST PART 3: VALIDATION (Day 2)
    def test_validation(self):
        self.section("PART 3: TESTING VALIDATION (Day 2)")

        validators_file = Path("backend/utils/validators.py")
        self.test(
            "Validators file exists",
            validators_file.exists(),
            "backend/utils/validators.py not found"
        )

        if validators_file.exists():
            with open(validators_file, 'r') as f:
                validators_content = f.read()

                self.test(
                    "Validators has validate_date_range",
                    "def validate_date_range" in validators_content,
                    "validate_date_range function not found"
                )
                self.test(
                    "Validators has validate_sku",
                    "def validate_sku" in validators_content,
                    "validate_sku function not found"
                )
                self.test(
                    "Validators has validate_transaction_type",
                    "def validate_transaction_type" in validators_content,
                    "validate_transaction_type function not found"
                )
                self.test(
                    "Validators has validate_city_name",
                    "def validate_city_name" in validators_content,
                    "validate_city_name function not found"
                )
                self.test(
                    "Validators imports from config",
                    "from core.config import" in validators_content,
                    "Validators not using config constants"
                )

        # Check sanitizers file exists
        sanitizers_file = Path("backend/utils/sanitizers.py")
        self.test(
            "Sanitizers file exists",
            sanitizers_file.exists(),
            "backend/utils/sanitizers.py not found"
        )

        if sanitizers_file.exists():
            with open(sanitizers_file, 'r') as f:
                sanitizers_content = f.read()
                self.test(
                    "Sanitizers has sanitize_string",
                    "def sanitize_string" in sanitizers_content,
                    "sanitize_string function not found"
                )
                self.test(
                    "Sanitizers has sanitize_date_string",
                    "def sanitize_date_string" in sanitizers_content,
                    "sanitize_date_string function not found"
                )

    # TEST PART 4: API ERROR HANDLING (Day 1)
    def test_api_error_handling(self):
        self.section("PART 4: TESTING API ERROR HANDLING (Day 1)")

        if not REQUESTS_AVAILABLE:
            print(f"{YELLOW}⚠ Skipping API tests (requests library not installed){RESET}")
            self.test(
                "Requests library installed",
                False,
                "Install with: pip install requests"
            )
            return

        # Test 1: Check if backend is running
        try:
            response = requests.get(
                f"{self.base_url}/api/health",
                timeout=2
            )
            self.backend_running = True
            self.test(
                "Backend is running",
                True,
                ""
            )
        except requests.exceptions.ConnectionError:
            print(f"{RED}⚠ WARNING: Backend not running at {self.base_url}{RESET}")
            print(f"{YELLOW}  Start backend: cd backend && python main.py{RESET}\n")
            self.test(
                "Backend is running",
                False,
                "Start backend: cd backend && python main.py"
            )
            return
        except Exception as e:
            self.test("Backend connectivity", False, str(e))
            return

        # Test 2: Valid request
        try:
            response = requests.get(
                f"{self.base_url}/api/metrics/trend",
                params={
                    "start_date": "2025-07-01",
                    "end_date": "2025-07-31",
                    "group_by": "day"
                },
                timeout=5
            )
            self.test(
                "Valid request returns success (200)",
                response.status_code == 200,
                f"Got status {response.status_code} instead of 200"
            )
        except Exception as e:
            self.test("Valid request succeeds", False, str(e))

        # Test 3: Invalid date range (reversed)
        try:
            response = requests.get(
                f"{self.base_url}/api/metrics/trend",
                params={
                    "start_date": "2025-07-31",
                    "end_date": "2025-07-01",
                    "group_by": "day"
                },
                timeout=5
            )
            self.test(
                "Invalid date range rejected (400 status)",
                response.status_code == 400,
                f"Got status {response.status_code}, expected 400"
            )
            if response.status_code == 400:
                try:
                    error_data = response.json()
                    has_error = "detail" in error_data or "error" in str(error_data).lower() or "message" in error_data
                    self.test(
                        "Invalid request has error message",
                        has_error,
                        f"Response: {response.text[:200]}"
                    )
                except:
                    self.test(
                        "Invalid request has error message",
                        "error" in response.text.lower() or "invalid" in response.text.lower(),
                        f"Response: {response.text[:200]}"
                    )
        except Exception as e:
            self.test("Invalid date range test", False, str(e))

        # Test 4: Missing required parameters
        try:
            response = requests.get(
                f"{self.base_url}/api/metrics/trend",
                timeout=5
            )
            # FastAPI returns 422 for missing required params
            self.test(
                "Missing params returns error (422 or 400)",
                response.status_code in [400, 422],
                f"Got status {response.status_code}, expected 400 or 422"
            )
        except Exception as e:
            self.test("Missing params test", False, str(e))

        # Test 5: Invalid transaction type
        try:
            response = requests.get(
                f"{self.base_url}/api/metrics/",
                params={
                    "start_date": "2025-07-01",
                    "end_date": "2025-07-31",
                    "transaction_type": "InvalidType"
                },
                timeout=5
            )
            # Should either reject or handle gracefully
            self.test(
                "Invalid transaction type handled",
                response.status_code in [200, 400, 422],
                f"Got unexpected status {response.status_code}"
            )
        except Exception as e:
            self.test("Invalid transaction type test", False, str(e))

        # Test 6: System still works after errors
        try:
            response = requests.get(
                f"{self.base_url}/api/metrics/trend",
                params={
                    "start_date": "2025-07-01",
                    "end_date": "2025-07-31",
                    "group_by": "day"
                },
                timeout=5
            )
            self.test(
                "System still works after errors",
                response.status_code == 200,
                "Backend crashed after error handling"
            )
        except Exception as e:
            self.test("System stability test", False, str(e))

    # TEST PART 5: ERROR HANDLING FUNCTIONS (Day 1)
    def test_error_handling_functions(self):
        self.section("PART 5: TESTING ERROR HANDLING FUNCTIONS (Day 1)")

        # Check error_handler file exists
        error_handler_file = Path("backend/utils/error_handler.py")
        self.test(
            "Error handler file exists",
            error_handler_file.exists(),
            "backend/utils/error_handler.py not found"
        )

        if error_handler_file.exists():
            with open(error_handler_file, 'r') as f:
                content = f.read()
                self.test(
                    "Error handler has format_error_response",
                    "def format_error_response" in content,
                    "format_error_response function not found"
                )
                self.test(
                    "Error handler has log_error",
                    "def log_error" in content,
                    "log_error function not found"
                )

        # Check services have error handling
        services_file = Path("backend/services/metrics_service.py")
        if services_file.exists():
            with open(services_file, 'r') as f:
                content = f.read()

                self.test(
                    "Services have try-except blocks",
                    content.count("try:") > 0 and content.count("except") > 0,
                    "No try-except blocks found in service files"
                )
                self.test(
                    "Services use error_handler",
                    "from utils.error_handler import" in content or "error_handler" in content,
                    "Services not using error_handler utility"
                )

        # Check API routes have error handling
        routes_file = Path("backend/api/routes/metrics.py")
        if routes_file.exists():
            with open(routes_file, 'r') as f:
                content = f.read()
                self.test(
                    "API routes have try-except blocks",
                    content.count("try:") > 0 and content.count("except") > 0,
                    "No try-except blocks found in API routes"
                )
                self.test(
                    "API routes use HTTPException",
                    "HTTPException" in content,
                    "API routes not using HTTPException for errors"
                )
                self.test(
                    "API routes use validation",
                    "validate_date_range" in content or "from utils.validators import" in content,
                    "API routes not using validation"
                )
                self.test(
                    "API routes use sanitization",
                    "sanitize" in content or "from utils.sanitizers import" in content,
                    "API routes not using sanitization"
                )

    def run_all_tests(self):
        """Run all verification tests"""
        print(f"\n{BLUE}{'='*60}{RESET}")
        print(f"{BLUE}CLEANUP SPRINT VERIFICATION{RESET}")
        print(f"{BLUE}Testing Days 1, 2, 3 Implementation{RESET}")
        print(f"{BLUE}{'='*60}{RESET}\n")

        self.test_logging()
        self.test_config()
        self.test_validation()
        self.test_api_error_handling()
        self.test_error_handling_functions()

        # Summary
        self.section("VERIFICATION SUMMARY")
        total = self.passed + self.failed
        percentage = (self.passed / total * 100) if total > 0 else 0

        print(f"Total Tests: {total}")
        print(f"{GREEN}Passed: {self.passed}{RESET}")
        print(f"{RED}Failed: {self.failed}{RESET}")
        print(f"Success Rate: {percentage:.1f}%\n")

        if self.failed == 0:
            print(f"{GREEN}{'='*60}{RESET}")
            print(f"{GREEN}✓ CLEANUP SPRINT COMPLETE!{RESET}")
            print(f"{GREEN}All verifications passed!{RESET}")
            print(f"{GREEN}{'='*60}{RESET}\n")
            return True
        else:
            print(f"{RED}{'='*60}{RESET}")
            print(f"{RED}✗ Some tests failed. Review above.{RESET}")
            print(f"{RED}{'='*60}{RESET}\n")
            return False


if __name__ == "__main__":
    # Change to project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    os.chdir(project_root)

    verifier = CleanupVerifier()
    success = verifier.run_all_tests()
    sys.exit(0 if success else 1)

