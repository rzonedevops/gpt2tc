"""Tests for gpt2tc package."""

import pytest
import tempfile
import os
from pathlib import Path

# Test data
SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog. This is a test sentence for compression."
SAMPLE_PROMPT = "Once upon a time"