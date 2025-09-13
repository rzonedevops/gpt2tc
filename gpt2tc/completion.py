"""GPT-2 Text Completion module."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class GPT2TextCompletion:
    """Text completion using GPT-2 model."""
    
    def __init__(self, model_name: str = "gpt2", device: Optional[str] = None):
        """
        Initialize the GPT-2 text completion model.
        
        Args:
            model_name: GPT-2 model variant to use (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
            device: Device to run the model on (cuda, cpu, or auto-detect)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading GPT-2 model: {model_name} on device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
    
    def complete_text(
        self,
        prompt: str,
        max_length: int = 100,
        num_return_sequences: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None
    ) -> Union[str, List[str]]:
        """
        Complete text using GPT-2 model.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            num_return_sequences: Number of sequences to return
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            pad_token_id: Padding token ID
            
        Returns:
            Completed text(s)
        """
        # Encode input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Set pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        
        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode outputs
        completed_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            completed_texts.append(text)
        
        # Return single string if only one sequence requested
        if num_return_sequences == 1:
            return completed_texts[0]
        
        return completed_texts
    
    def get_perplexity(self, text: str) -> float:
        """
        Calculate perplexity of the given text.
        
        Args:
            text: Input text
            
        Returns:
            Perplexity score
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss)
        
        return perplexity.item()
    
    def score_text(self, text: str) -> float:
        """
        Score text likelihood using the model.
        
        Args:
            text: Input text to score
            
        Returns:
            Log likelihood score
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        
        return -loss.item()  # Return negative loss as likelihood score