# Tests Directory

This directory contains all test scripts for the vggt-dataset-builder project.

## Running Tests

### Using pytest (recommended)

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov

# Run a specific test file
pytest tests/test_bidirectional.py

# Run a specific test function
pytest tests/test_bidirectional.py::test_file_detection
```

### Running as scripts

Each test file can also be run directly as a Python script:

```bash
python tests/test_bidirectional.py
python tests/test_triplet_detection.py
```

## Test Files

- **test_bidirectional.py** - Tests for bidirectional file detection logic. Verifies that the system correctly identifies forward and reverse direction file pairs.

- **test_triplet_detection.py** - Tests for triplet detection logic. Verifies that complete image triplets (splats, reference, target) are correctly identified in scene directories.

## Adding New Tests

1. Create a new test file in this directory following the naming convention `test_*.py`
2. Define test functions prefixed with `test_`
3. Run with `pytest` or as a script
4. Optionally mark tests with pytest markers:
   - `@pytest.mark.unit` for unit tests
   - `@pytest.mark.integration` for integration tests

Example:

```python
import pytest

@pytest.mark.unit
def test_example():
    assert True
```

## Test Configuration

The pytest configuration is defined in `pyproject.toml` at the workspace root:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_functions = ["test_*"]
```
