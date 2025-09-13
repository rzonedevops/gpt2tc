"""GPT-2 Text Compression module."""

import gzip
import json
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


class GPT2TextCompression:
    """Text compression using GPT-2 model for context-aware compression."""
    
    def __init__(self, model_name: str = "gpt2", device: Optional[str] = None):
        """
        Initialize the GPT-2 text compression model.
        
        Args:
            model_name: GPT-2 model variant to use
            device: Device to run the model on
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading GPT-2 model for compression: {model_name} on device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
    
    def compress_text_simple(self, text: str) -> bytes:
        """
        Simple compression using gzip.
        
        Args:
            text: Input text to compress
            
        Returns:
            Compressed bytes
        """
        return gzip.compress(text.encode('utf-8'))
    
    def decompress_text_simple(self, compressed_data: bytes) -> str:
        """
        Simple decompression using gzip.
        
        Args:
            compressed_data: Compressed bytes
            
        Returns:
            Decompressed text
        """
        return gzip.decompress(compressed_data).decode('utf-8')
    
    def compress_text_adaptive(self, text: str, context_window: int = 512) -> Dict:
        """
        Adaptive compression using GPT-2 predictions for better compression.
        
        Args:
            text: Input text to compress
            context_window: Size of context window for predictions
            
        Returns:
            Dictionary containing compressed data and metadata
        """
        # Tokenize the text
        tokens = self.tokenizer.encode(text)
        
        compressed_tokens = []
        predictions = []
        
        # Process text in chunks
        for i in range(0, len(tokens), context_window):
            chunk = tokens[i:i + context_window]
            
            if len(chunk) < context_window and i > 0:
                # For smaller chunks, use previous context for prediction
                context = tokens[max(0, i - context_window):i]
                context_tensor = torch.tensor([context]).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(context_tensor)
                    predicted_probs = torch.softmax(outputs.logits[0, -1, :], dim=0)
                
                # Store top predictions for this chunk
                top_k = 10
                top_probs, top_indices = torch.topk(predicted_probs, top_k)
                predictions.append({
                    'probs': top_probs.cpu().numpy().tolist(),
                    'indices': top_indices.cpu().numpy().tolist()
                })
            
            compressed_tokens.extend(chunk)
        
        # Create compression metadata
        compression_data = {
            'tokens': compressed_tokens,
            'predictions': predictions,
            'vocab_size': self.tokenizer.vocab_size,
            'model_name': self.model_name,
            'original_length': len(text)
        }
        
        return compression_data
    
    def decompress_text_adaptive(self, compression_data: Dict) -> str:
        """
        Decompress text using adaptive compression data.
        
        Args:
            compression_data: Compression data dictionary
            
        Returns:
            Decompressed text
        """
        tokens = compression_data['tokens']
        decoded_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return decoded_text
    
    def get_compression_ratio(self, original_text: str, compressed_data: Union[bytes, Dict]) -> float:
        """
        Calculate compression ratio.
        
        Args:
            original_text: Original text
            compressed_data: Compressed data (bytes or dict)
            
        Returns:
            Compression ratio (original_size / compressed_size)
        """
        original_size = len(original_text.encode('utf-8'))
        
        if isinstance(compressed_data, bytes):
            compressed_size = len(compressed_data)
        else:
            # For dict compression, estimate size
            compressed_bytes = pickle.dumps(compressed_data)
            compressed_size = len(compressed_bytes)
        
        return original_size / compressed_size if compressed_size > 0 else 0
    
    def compress_with_entropy_coding(self, text: str) -> Dict:
        """
        Compress text using entropy coding based on GPT-2 predictions.
        
        Args:
            text: Input text to compress
            
        Returns:
            Compressed data with entropy coding
        """
        tokens = self.tokenizer.encode(text)
        
        # Calculate token probabilities using the model
        input_tensor = torch.tensor([tokens]).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor, labels=input_tensor)
            logits = outputs.logits
        
        # Calculate entropy for each token
        entropies = []
        for i in range(len(tokens) - 1):
            probs = torch.softmax(logits[0, i, :], dim=0)
            token_prob = probs[tokens[i + 1]]
            entropy = -torch.log2(token_prob)
            entropies.append(entropy.item())
        
        # Create compressed representation
        compression_data = {
            'tokens': tokens,
            'entropies': entropies,
            'total_entropy': sum(entropies),
            'model_name': self.model_name,
            'original_length': len(text)
        }
        
        return compression_data
    
    def analyze_text_compressibility(self, text: str) -> Dict:
        """
        Analyze how compressible a text is using GPT-2.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Analysis results
        """
        # Get simple compression ratio
        simple_compressed = self.compress_text_simple(text)
        simple_ratio = self.get_compression_ratio(text, simple_compressed)
        
        # Get entropy-based analysis
        entropy_data = self.compress_with_entropy_coding(text)
        
        # Calculate perplexity
        tokens = self.tokenizer.encode(text)
        input_tensor = torch.tensor([tokens]).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor, labels=input_tensor)
            perplexity = torch.exp(outputs.loss).item()
        
        return {
            'simple_compression_ratio': simple_ratio,
            'total_entropy_bits': entropy_data['total_entropy'],
            'average_entropy_per_token': entropy_data['total_entropy'] / len(entropy_data['entropies']) if entropy_data['entropies'] else 0,
            'perplexity': perplexity,
            'token_count': len(tokens),
            'character_count': len(text),
            'compressibility_score': simple_ratio * (1 / perplexity) if perplexity > 0 else 0
        }