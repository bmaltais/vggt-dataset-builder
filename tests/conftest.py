"""Pytest configuration and fixtures for vggt-dataset-builder tests."""

import sys
from pathlib import Path

# Add the workspace root to sys.path so tests can import modules
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root))

import pytest

@pytest.fixture
def workspace_root_path():
    """Provide the workspace root path to tests."""
    return Path(__file__).parent.parent

@pytest.fixture
def test_data_dir():
    """Provide the test data directory path."""
    return Path(__file__).parent / "data"
