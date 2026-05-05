import base64
import pytest

"""Base test class and shared assertion utilities for the test suite.

This module provides common validation helpers and assertion methods used across
all test suites to ensure consistent test behavior and error reporting.
"""
class BaseTest:
    """
    Base class for all test suites.
    Contains shared assertions and helpers - do not add domain logic here.
    """
    def assert_valid_base64(self, value: str) -> None:
        """Verifies that a string is valid Base64."""
        try:
            base64.b64decode(value, validate=True)
        except Exception:
            pytest.fail(f"Invalid Base64 string: {value[:40]}…")

    def assert_is_ratio(self, value: float) -> None:
        """Verifies that a value is within [0.0, 1.0]."""
        assert isinstance(value, float), f"Expected float, got: {type(value)}"
        assert 0.0 <= value <= 1.0, f"Value {value} is outside the allowed range [0, 1]."

    def assert_is_bool(self, value, field_name: str = "field") -> None:
        """Verifies bool type - JSON can sometimes return 0/1 instead of true/false."""
        assert isinstance(value, bool), f"'{field_name}' should be bool, got: {type(value)}"
        
    def assert_response_keys(self, data: dict, expected_keys: set) -> None:
        """Verifies that the response contains all required keys."""
        missing = expected_keys - data.keys()
        assert not missing, f"Missing keys in response: {missing}"

    def assert_glaucoma_response(self, data: dict) -> None:
        """
        Complete validation of the /analyze-glaucoma response.
        Call this in API tests instead of writing each assertion manually.
        """
        self.assert_response_keys(
            data,
            {"has_glaucoma", "confidence", "cup_to_disc_ratio", "image_base64"}
        )
        self.assert_is_bool(data["has_glaucoma"], "has_glaucoma")
        self.assert_is_ratio(data["confidence"])
        self.assert_is_ratio(data["cup_to_disc_ratio"])
        self.assert_valid_base64(data["image_base64"])