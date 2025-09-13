# gpt2tc

Text completion and compression using GPT-2

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

A Python package for text completion and compression using GPT-2 models. This package provides both programmatic API and command-line interface for text generation and various compression techniques.

## Features

- **Text Completion**: Generate text continuations using GPT-2 models
- **Text Compression**: Multiple compression methods including simple, adaptive, and entropy-based compression
- **Text Analysis**: Analyze text compressibility and linguistic properties
- **CLI Interface**: Easy-to-use command-line tools
- **Multiple GPT-2 Models**: Support for different GPT-2 model sizes
- **Flexible Output**: JSON, text, and binary output formats

## Installation

Install the package using pip:

```bash
pip install gpt2tc
```

Or install from source:

```bash
git clone https://github.com/rzonedevops/gpt2tc.git
cd gpt2tc
pip install -e .
```

For development with all dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Command Line Usage

**Text Completion:**
```bash
# Basic text completion
gpt2tc complete "The future of artificial intelligence is"

# Advanced completion with parameters
gpt2tc complete "Once upon a time" --max-length 200 --temperature 0.8 --num-sequences 3
```

**Text Compression:**
```bash
# Simple compression
gpt2tc compress "Your text here" --method simple --output compressed.gz

# Adaptive compression using GPT-2 context
gpt2tc compress "Your text here" --method adaptive --output compressed.json

# Entropy-based analysis
gpt2tc compress "Your text here" --method entropy --output entropy.json
```

**Text Analysis:**
```bash
# Analyze text properties
gpt2tc analyze "Your text here"

# Analyze from file
gpt2tc analyze --file input.txt
```

### Python API Usage

**Text Completion:**
```python
from gpt2tc import GPT2TextCompletion

# Initialize the model
completion = GPT2TextCompletion(model_name="gpt2")

# Generate text
result = completion.complete_text(
    prompt="The future of artificial intelligence is",
    max_length=100,
    temperature=0.8
)
print(result)

# Generate multiple sequences
results = completion.complete_text(
    prompt="Once upon a time",
    max_length=150,
    num_return_sequences=3,
    temperature=0.9
)
for i, text in enumerate(results):
    print(f"Story {i+1}: {text}")
```

**Text Compression:**
```python
from gpt2tc import GPT2TextCompression

# Initialize the compression model
compression = GPT2TextCompression(model_name="gpt2")

# Simple compression
text = "Your long text here..."
compressed = compression.compress_text_simple(text)
decompressed = compression.decompress_text_simple(compressed)

# Adaptive compression
adaptive_compressed = compression.compress_text_adaptive(text)
adaptive_decompressed = compression.decompress_text_adaptive(adaptive_compressed)

# Analyze compressibility
analysis = compression.analyze_text_compressibility(text)
print(f"Compression ratio: {analysis['simple_compression_ratio']:.2f}")
print(f"Perplexity: {analysis['perplexity']:.2f}")
```

## CLI Commands

### `gpt2tc complete`
Generate text completions using GPT-2.

**Options:**
- `--max-length, -l`: Maximum length of generated text (default: 100)
- `--num-sequences, -n`: Number of sequences to generate (default: 1)
- `--temperature, -t`: Sampling temperature (default: 1.0)
- `--top-k`: Top-k sampling parameter (default: 50)
- `--top-p`: Top-p nucleus sampling parameter (default: 0.95)
- `--output, -o`: Output file path

### `gpt2tc compress`
Compress text using various methods.

**Options:**
- `--file, -f`: Input file path
- `--output, -o`: Output file path
- `--method`: Compression method (simple, adaptive, entropy)

### `gpt2tc decompress`
Decompress previously compressed text.

**Options:**
- `--output, -o`: Output file path
- `--method`: Decompression method (simple, adaptive)

### `gpt2tc analyze`
Analyze text properties and compressibility.

**Options:**
- `--file, -f`: Input file path

## Global Options

- `--model, -m`: GPT-2 model to use (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
- `--verbose, -v`: Enable verbose output

## Models

The package supports different GPT-2 model variants:

- `gpt2`: 124M parameters (default)
- `gpt2-medium`: 355M parameters
- `gpt2-large`: 774M parameters
- `gpt2-xl`: 1.5B parameters

Larger models provide better quality but require more computational resources.

## Examples

### Example 1: Story Generation
```bash
gpt2tc complete "In a world where robots have emotions," \
  --max-length 300 \
  --temperature 0.8 \
  --num-sequences 2 \
  --output stories.txt
```

### Example 2: Code Completion
```bash
gpt2tc complete "def fibonacci(n):" \
  --model gpt2-medium \
  --max-length 150 \
  --temperature 0.3
```

### Example 3: Text Compression Analysis
```bash
# Compress a file and analyze compression ratio
gpt2tc compress --file document.txt --method adaptive --output compressed.json
gpt2tc analyze --file document.txt
```

### Example 4: Batch Processing
```python
from gpt2tc import GPT2TextCompletion, GPT2TextCompression
import json

# Initialize models
completion = GPT2TextCompletion("gpt2-medium")
compression = GPT2TextCompression("gpt2-medium")

# Process multiple texts
texts = ["Prompt 1", "Prompt 2", "Prompt 3"]
results = []

for prompt in texts:
    # Generate completion
    completed = completion.complete_text(prompt, max_length=100)
    
    # Analyze compression
    analysis = compression.analyze_text_compressibility(completed)
    
    results.append({
        "prompt": prompt,
        "completion": completed,
        "compression_ratio": analysis["simple_compression_ratio"],
        "perplexity": analysis["perplexity"]
    })

# Save results
with open("batch_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- NumPy
- Click
- tqdm

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please use the GitHub issue tracker.
