"""Tests for text completion functionality."""

import pytest
from unittest.mock import Mock, patch
from gpt2tc.completion import GPT2TextCompletion

class TestGPT2TextCompletion:
    """Test cases for GPT2TextCompletion class."""
    
    @patch('gpt2tc.completion.GPT2LMHeadModel')
    @patch('gpt2tc.completion.GPT2Tokenizer')
    def test_init(self, mock_tokenizer, mock_model):
        """Test initialization of GPT2TextCompletion."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        
        # Mock model
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Test initialization
        completion = GPT2TextCompletion(model_name="gpt2")
        
        assert completion.model_name == "gpt2"
        assert completion.device in ["cuda", "cpu"]
        mock_tokenizer.from_pretrained.assert_called_once_with("gpt2")
        mock_model.from_pretrained.assert_called_once_with("gpt2")
    
    def test_init_with_custom_device(self):
        """Test initialization with custom device."""
        with patch('gpt2tc.completion.GPT2LMHeadModel'), \
             patch('gpt2tc.completion.GPT2Tokenizer'):
            completion = GPT2TextCompletion(model_name="gpt2", device="cpu")
            assert completion.device == "cpu"
    
    @patch('gpt2tc.completion.torch')
    def test_device_selection(self, mock_torch):
        """Test automatic device selection."""
        # Test CUDA available
        mock_torch.cuda.is_available.return_value = True
        with patch('gpt2tc.completion.GPT2LMHeadModel'), \
             patch('gpt2tc.completion.GPT2Tokenizer'):
            completion = GPT2TextCompletion(model_name="gpt2")
            assert completion.device == "cuda"
        
        # Test CUDA not available
        mock_torch.cuda.is_available.return_value = False
        with patch('gpt2tc.completion.GPT2LMHeadModel'), \
             patch('gpt2tc.completion.GPT2Tokenizer'):
            completion = GPT2TextCompletion(model_name="gpt2")
            assert completion.device == "cpu"
    
    @patch('gpt2tc.completion.torch')
    @patch('gpt2tc.completion.GPT2LMHeadModel')
    @patch('gpt2tc.completion.GPT2Tokenizer')
    def test_complete_text_basic(self, mock_tokenizer, mock_model, mock_torch):
        """Test basic text completion."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer_instance.eos_token_id = 50256
        mock_tokenizer_instance.encode.return_value = [1, 2, 3]
        mock_tokenizer_instance.decode.return_value = "Completed text"
        
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Mock tensor operations
        mock_tensor = Mock()
        mock_torch.tensor.return_value = mock_tensor
        mock_tensor.to.return_value = mock_tensor
        
        # Mock model generate
        mock_output = Mock()
        mock_model_instance.generate.return_value = [mock_tensor]
        
        # Test completion
        completion = GPT2TextCompletion(model_name="gpt2")
        result = completion.complete_text("Test prompt")
        
        assert result == "Completed text"
        mock_tokenizer_instance.encode.assert_called_once_with("Test prompt", return_tensors="pt")
        mock_model_instance.generate.assert_called_once()
    
    def test_model_name_validation(self):
        """Test that different model names are accepted."""
        valid_models = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
        
        for model_name in valid_models:
            with patch('gpt2tc.completion.GPT2LMHeadModel'), \
                 patch('gpt2tc.completion.GPT2Tokenizer'):
                completion = GPT2TextCompletion(model_name=model_name)
                assert completion.model_name == model_name