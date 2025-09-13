"""Tests for CLI functionality."""

import pytest
from click.testing import CliRunner
from unittest.mock import patch
from gpt2tc.cli import main

class TestCLI:
    """Test cases for CLI functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_version_command(self):
        """Test version command."""
        result = self.runner.invoke(main, ['version'])
        assert result.exit_code == 0
        assert 'gpt2tc version' in result.output
    
    def test_help_command(self):
        """Test help command."""
        result = self.runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert 'GPT-2 Text Completion and Compression tool' in result.output
    
    @patch('gpt2tc.cli.GPT2TextCompletion')
    def test_complete_command(self, mock_completion_class):
        """Test complete command."""
        # Mock the completion class
        mock_completion = mock_completion_class.return_value
        mock_completion.complete_text.return_value = "This is a completed text."
        
        result = self.runner.invoke(main, ['complete', 'Test prompt'])
        
        assert result.exit_code == 0
        assert 'Generating text completion' in result.output
        assert 'This is a completed text.' in result.output
    
    @patch('gpt2tc.cli.GPT2TextCompression')
    def test_compress_command_simple(self, mock_compression_class):
        """Test compress command with simple method."""
        # Mock the compression class
        mock_compression = mock_compression_class.return_value
        mock_compression.compress_text_simple.return_value = b'compressed_data'
        mock_compression.get_compression_ratio.return_value = 2.5
        
        result = self.runner.invoke(main, ['compress', 'Test text', '--method', 'simple'])
        
        assert result.exit_code == 0
        assert 'Compressing text using method: simple' in result.output
    
    @patch('gpt2tc.cli.GPT2TextCompression')
    def test_analyze_command(self, mock_compression_class):
        """Test analyze command."""
        # Mock the compression class
        mock_compression = mock_compression_class.return_value
        mock_compression.analyze_text_compressibility.return_value = {
            'simple_compression_ratio': 2.5,
            'total_entropy_bits': 100.0,
            'average_entropy_per_token': 5.0,
            'perplexity': 15.5,
            'token_count': 20,
            'character_count': 100,
            'compressibility_score': 0.8
        }
        
        with patch('gpt2tc.cli.GPT2TextCompletion') as mock_completion_class:
            mock_completion = mock_completion_class.return_value
            mock_completion.get_perplexity.return_value = 15.5
            
            result = self.runner.invoke(main, ['analyze', 'Test text'])
            
            assert result.exit_code == 0
            assert 'Analyzing text' in result.output
            assert 'Simple compression ratio: 2.50' in result.output
    
    def test_invalid_method(self):
        """Test invalid compression method."""
        result = self.runner.invoke(main, ['compress', 'Test text', '--method', 'invalid'])
        
        assert result.exit_code != 0
        assert 'Invalid value' in result.output