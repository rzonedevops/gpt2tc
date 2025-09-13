"""
gpt2tc: Text completion and compression using GPT-2

A Python package for text completion and compression using GPT-2 models.
"""

__version__ = "0.1.0"
__author__ = "rzonedevops"
__email__ = ""
__license__ = "AGPL-3.0"

from .completion import GPT2TextCompletion
from .compression import GPT2TextCompression

__all__ = ["GPT2TextCompletion", "GPT2TextCompression"]