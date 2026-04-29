import base64
import pytest

class BaseTest:
    """
    Base class for all test suites.
    Provides common validation helpers and ensures a consistent testing structure.
    """
    
    def assert_valid_base64(self, b64_string: str):
        """Verifies if a string is a valid Base64 encoded object."""
        try:
            # Check if padding is correct and can be decoded
            base64.b64decode(b64_string, validate=True)
        except Exception:
            pytest.fail("The response contains an invalid Base64 string.")

    def assert_is_ratio(self, value: float):
        """Checks if a numeric value is a valid ratio within [0, 1]."""
        assert 0.0 <= value <= 1.0, f"Value {value} is outside the allowed range [0, 1]."