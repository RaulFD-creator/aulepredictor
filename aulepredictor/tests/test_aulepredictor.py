"""
Unit and regression test for the aulepredictor package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import aulepredictor


def test_aulepredictor_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "aulepredictor" in sys.modules
