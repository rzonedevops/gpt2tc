#!/usr/bin/env python3
"""
Example usage of the gpt2tc package for text completion and compression.
This example demonstrates key features without requiring model downloads during development.
"""

import tempfile
import json
from pathlib import Path

def main():
    """Demonstrate gpt2tc package functionality."""
    
    print("🚀 GPT2TC Package Example")
    print("=" * 50)
    
    # Import the package
    try:
        from gpt2tc import GPT2TextCompletion, GPT2TextCompression
        print("✓ Successfully imported gpt2tc package")
    except ImportError as e:
        print(f"✗ Failed to import package: {e}")
        return
    
    # Example 1: Simple text compression
    print("\n📦 Example 1: Simple Text Compression")
    print("-" * 30)
    
    sample_text = """
    GPT-2 (Generative Pre-trained Transformer 2) is a large language model 
    developed by OpenAI. It is capable of generating human-like text and can 
    be used for various natural language processing tasks including text 
    completion, summarization, and compression analysis.
    """
    
    # Mock model loading for demonstration
    from unittest.mock import patch, Mock
    with patch('gpt2tc.compression.AutoTokenizer'), \
         patch('gpt2tc.compression.AutoModelForCausalLM'):
        
        compression = GPT2TextCompression(model_name="gpt2")
        
        # Simple compression
        compressed = compression.compress_text_simple(sample_text.strip())
        decompressed = compression.decompress_text_simple(compressed)
        ratio = compression.get_compression_ratio(sample_text.strip(), compressed)
        
        print(f"Original size: {len(sample_text.strip())} characters")
        print(f"Compressed size: {len(compressed)} bytes")
        print(f"Compression ratio: {ratio:.2f}")
        print(f"Decompression successful: {decompressed == sample_text.strip()}")
    
    # Example 2: CLI usage demonstration
    print("\n💻 Example 2: CLI Usage")
    print("-" * 30)
    
    print("Available CLI commands:")
    print("• gpt2tc complete 'Your prompt here' --max-length 100")
    print("• gpt2tc compress 'Your text' --method simple --output file.gz")
    print("• gpt2tc analyze 'Your text' # Analyze compressibility")
    print("• gpt2tc decompress file.gz --method simple")
    print("• gpt2tc version")
    
    # Example 3: File-based operations
    print("\n📄 Example 3: File Operations")
    print("-" * 30)
    
    # Create a temporary file for demonstration
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample_text.strip())
        temp_file = f.name
    
    print(f"Created temporary file: {Path(temp_file).name}")
    print("CLI usage for files:")
    print(f"• gpt2tc compress --file {temp_file} --method adaptive")
    print(f"• gpt2tc analyze --file {temp_file}")
    
    # Clean up
    Path(temp_file).unlink()
    
    # Example 4: Model variants
    print("\n🧠 Example 4: Model Variants")
    print("-" * 30)
    
    models = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
    print("Supported GPT-2 models:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    
    print("\nUsage with different models:")
    print("gpt2tc --model gpt2-medium complete 'Your prompt'")
    
    print("\n🎉 Package Features Summary:")
    print("-" * 30)
    print("✓ Text completion using GPT-2")
    print("✓ Multiple compression methods (simple, adaptive, entropy)")
    print("✓ Text analysis and compressibility scoring")
    print("✓ Command-line interface")
    print("✓ Python API for programmatic usage")
    print("✓ Support for multiple GPT-2 model sizes")
    print("✓ File and text input/output")
    print("✓ Configurable generation parameters")
    
    print("\n📚 Installation:")
    print("pip install gpt2tc")
    
    print("\n🔗 Repository:")
    print("https://github.com/rzonedevops/gpt2tc")

if __name__ == "__main__":
    main()