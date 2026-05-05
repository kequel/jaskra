# Test Writing Rules

Rules and conventions for writing tests in this repository.  
Read this before opening a test-related issue or submitting a PR that adds tests.


---

## Project Tests Structure

```
tests/
├── conftest.py           # shared fixtures (images, masks, mock pipeline outputs)
├── base_test.py          # BaseTest with shared assertion helpers
├── backend/
├── mobile/
├── ai_pipeline/
├── ai_yolo/
└── ai_unet/

```

Every subdirectory must have an `__init__.py` file.

---

## Naming Conventions

| Thing | Convention | Example |
|---|---|---|
| Test file | `test_<module>.py` | `test_pipeline.py` |
| Test class | `Test<Feature>` | `TestCalculateCDR` |
| Test method | `test_<what>_<expected_result>` | `test_cdr_empty_disc_returns_zero` |
| Fixture | `snake_case`, descriptive noun | `mock_pipeline_output_positive` |


---

## How to Write a Test Class

Every test class **must**:

1. Inherit from `BaseTest`
2. Have a class-level docstring that is the test plan for the entire class (see format below)
3. Have a method-level docstring per test method (same format, one test only)

```python
import pytest
from unittest.mock import patch
from tests.base_test import BaseTest


class TestMyFeature(BaseTest):
    """
    <Short description of what this class tests>
    """

    def test_something_returns_expected(self):
        """
        TEST NAME: test_something_returns_expected
        COMPONENT: SomeClass.some_method()
        1. Create input with known properties.
        2. Call some_method().
        3. Assert the return value matches expected.
        """
        result = SomeClass().some_method(input_value)
        assert result == expected_value
```

---

## Docstring Format

The docstring format is **mandatory** to keep tests reviewable.

```
TEST NAME: <exact method name>
COMPONENT: <what part of the system this tests, e.g. "POST /analyze-glaucoma", "GlaucomaPipeline.calculate_cdr()">
1. <First step — what you set up or mock>
2. <Second step — what you call>
3. <Third step — what you assert>
```

Rules:
- **Minimum 2 steps, maximum 6.** If you need more, split into multiple tests.
- Steps are **actions**, not observations. Write "Assert X equals Y", not "X should equal Y".
- The `COMPONENT` field refers to the production code being exercised, not the test file.

---

## Fixtures and Mocks

All shared fixtures live in `tests/conftest.py`. **Do not define fixtures inside test files**.

### Patching the pipeline

The pipeline is imported dynamically inside the endpoint functions in `main.py`. Patch it at its definition location:

```python
PIPELINE_PATCH = "pipeline.pipeline.GlaucomaPipeline.run"

@patch(PIPELINE_PATCH)
def test_something(self, mock_run, sample_image, mock_pipeline_output_positive):
    mock_run.return_value = mock_pipeline_output_positive
    ...
```

---

## Running Tests Locally

```bash
# install dependencies
pip install -r backend/requirements.txt pytest pytest-cov httpx Pillow

# integration tests (start server first)
uvicorn backend/main:app --host 0.0.0.0 --port 8000 &

# single file
pytest tests/backend/test_api.py -v
```
