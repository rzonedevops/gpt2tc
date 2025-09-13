"""Tests for text compression functionality."""

import pytest
import gzip
from unittest.mock import Mock, patch
from gpt2tc.compression import GPT2TextCompression

class TestGPT2TextCompression:
    """Test cases for GPT2TextCompression class."""
    
    @patch('gpt2tc.compression.GPT2LMHeadModel')
    @patch('gpt2tc.compression.GPT2Tokenizer')
    def test_init(self, mock_tokenizer, mock_model):
        """Test initialization of GPT2TextCompression."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        
        # Mock model
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Test initialization
        compression = GPT2TextCompression(model_name="gpt2")
        
        assert compression.model_name == "gpt2"
        assert compression.device in ["cuda", "cpu"]
        mock_tokenizer.from_pretrained.assert_called_once_with("gpt2")
        mock_model.from_pretrained.assert_called_once_with("gpt2")
    
    def test_simple_compression_decompression(self):
        """Test simple compression and decompression."""
        with patch('gpt2tc.compression.GPT2LMHeadModel'), \
             patch('gpt2tc.compression.GPT2Tokenizer'):
            compression = GPT2TextCompression(model_name="gpt2")
            
            test_text = "This is a test string for compression."
            
            # Test compression
            compressed = compression.compress_text_simple(test_text)
            assert isinstance(compressed, bytes)
            assert len(compressed) > 0
            
            # Test decompression
            decompressed = compression.decompress_text_simple(compressed)
            assert decompressed == test_text
    
    def test_compression_ratio_calculation(self):
        """Test compression ratio calculation."""
        with patch('gpt2tc.compression.GPT2LMHeadModel'), \
             patch('gpt2tc.compression.GPT2Tokenizer'):
            compression = GPT2TextCompression(model_name="gpt2")
            
            test_text = "This is a test string for compression ratio calculation."
            compressed = compression.compress_text_simple(test_text)
            
            ratio = compression.get_compression_ratio(test_text, compressed)
            assert isinstance(ratio, float)
            assert ratio > 0
    
    @patch('gpt2tc.compression.torch')
    @patch('gpt2tc.compression.GPT2LMHeadModel')
    @patch('gpt2tc.compression.GPT2Tokenizer')
    def test_adaptive_compression(self, mock_tokenizer, mock_model, mock_torch):
        """Test adaptive compression using GPT-2 context."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer_instance.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer_instance.vocab_size = 50257
        mock_tokenizer_instance.decode.return_value = "Test text"
        
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Mock tensor operations
        mock_tensor = Mock()
        mock_torch.tensor.return_value = mock_tensor
        mock_tensor.to.return_value = mock_tensor
        
        # Mock model output
        mock_outputs = Mock()
        mock_outputs.logits = mock_torch.randn(1, 5, 50257)
        mock_model_instance.return_value = mock_outputs
        
        # Mock torch operations
        mock_torch.softmax.return_value = mock_torch.randn(50257)
        mock_torch.topk.return_value = (mock_torch.randn(10), mock_torch.randint(0, 50257, (10,)))
        
        # Test adaptive compression
        compression = GPT2TextCompression(model_name="gpt2")
        test_text = "This is a test for adaptive compression."
        
        result = compression.compress_text_adaptive(test_text)
        
        assert isinstance(result, dict)
        assert 'tokens' in result
        assert 'model_name' in result
        assert 'original_length' in result
        assert result['model_name'] == "gpt2"
        assert result['original_length'] == len(test_text)
    
    def test_model_device_selection(self):
        """Test device selection for models."""
        with patch('gpt2tc.compression.torch') as mock_torch, \
             patch('gpt2tc.compression.GPT2LMHeadModel'), \
             patch('gpt2tc.compression.GPT2Tokenizer'):
            
            # Test CUDA available
            mock_torch.cuda.is_available.return_value = True
            compression = GPT2TextCompression(model_name="gpt2")
            assert compression.device == "cuda"
            
            # Test CUDA not available
            mock_torch.cuda.is_available.return_value = False
            compression = GPT2TextCompression(model_name="gpt2")
            assert compression.device == "cpu"
            
            # Test custom device
            compression = GPT2TextCompression(model_name="gpt2", device="cpu")
            assert compression.device == "cpu"