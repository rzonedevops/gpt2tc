"""Command Line Interface for gpt2tc."""

import click
import json
import sys
import logging
from pathlib import Path
from typing import Optional

from .completion import GPT2TextCompletion
from .compression import GPT2TextCompression

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--model', '-m', default='gpt2', help='GPT-2 model to use (gpt2, gpt2-medium, gpt2-large, gpt2-xl)')
@click.pass_context
def main(ctx, verbose, model):
    """GPT-2 Text Completion and Compression tool."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['model'] = model
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@main.command()
@click.argument('prompt', type=str)
@click.option('--max-length', '-l', default=100, help='Maximum length of generated text')
@click.option('--num-sequences', '-n', default=1, help='Number of sequences to generate')
@click.option('--temperature', '-t', default=1.0, help='Sampling temperature')
@click.option('--top-k', default=50, help='Top-k sampling parameter')
@click.option('--top-p', default=0.95, help='Top-p (nucleus) sampling parameter')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.pass_context
def complete(ctx, prompt, max_length, num_sequences, temperature, top_k, top_p, output):
    """Complete text using GPT-2."""
    try:
        model_name = ctx.obj['model']
        completion_model = GPT2TextCompletion(model_name=model_name)
        
        click.echo(f"Generating text completion with model: {model_name}")
        click.echo(f"Prompt: {prompt}")
        click.echo("-" * 50)
        
        completed_texts = completion_model.complete_text(
            prompt=prompt,
            max_length=max_length,
            num_return_sequences=num_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        if isinstance(completed_texts, str):
            completed_texts = [completed_texts]
        
        for i, text in enumerate(completed_texts):
            click.echo(f"Completion {i+1}:")
            click.echo(text)
            click.echo("-" * 50)
        
        # Save to file if specified
        if output:
            output_path = Path(output)
            with open(output_path, 'w', encoding='utf-8') as f:
                if num_sequences == 1:
                    f.write(completed_texts[0])
                else:
                    for i, text in enumerate(completed_texts):
                        f.write(f"=== Completion {i+1} ===\n")
                        f.write(text)
                        f.write("\n\n")
            click.echo(f"Output saved to: {output_path}")
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('text', type=str, required=False)
@click.option('--file', '-f', type=click.Path(exists=True), help='Input file path')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--method', default='simple', type=click.Choice(['simple', 'adaptive', 'entropy']), help='Compression method')
@click.pass_context
def compress(ctx, text, file, output, method):
    """Compress text using GPT-2."""
    try:
        # Get input text
        if file:
            with open(file, 'r', encoding='utf-8') as f:
                input_text = f.read()
        elif text:
            input_text = text
        else:
            click.echo("Error: Either provide text as argument or use --file option", err=True)
            sys.exit(1)
        
        model_name = ctx.obj['model']
        compression_model = GPT2TextCompression(model_name=model_name)
        
        click.echo(f"Compressing text using method: {method}")
        click.echo(f"Original length: {len(input_text)} characters")
        
        if method == 'simple':
            compressed_data = compression_model.compress_text_simple(input_text)
            ratio = compression_model.get_compression_ratio(input_text, compressed_data)
            
            click.echo(f"Compressed size: {len(compressed_data)} bytes")
            click.echo(f"Compression ratio: {ratio:.2f}")
            
            if output:
                with open(output, 'wb') as f:
                    f.write(compressed_data)
                click.echo(f"Compressed data saved to: {output}")
        
        elif method == 'adaptive':
            compressed_data = compression_model.compress_text_adaptive(input_text)
            ratio = compression_model.get_compression_ratio(input_text, compressed_data)
            
            click.echo(f"Compression ratio: {ratio:.2f}")
            click.echo(f"Tokens: {len(compressed_data['tokens'])}")
            
            if output:
                with open(output, 'w', encoding='utf-8') as f:
                    json.dump(compressed_data, f, indent=2)
                click.echo(f"Compressed data saved to: {output}")
        
        elif method == 'entropy':
            entropy_data = compression_model.compress_with_entropy_coding(input_text)
            
            click.echo(f"Total entropy: {entropy_data['total_entropy']:.2f} bits")
            click.echo(f"Average entropy per token: {entropy_data['total_entropy'] / len(entropy_data['entropies']):.2f} bits")
            
            if output:
                with open(output, 'w', encoding='utf-8') as f:
                    json.dump(entropy_data, f, indent=2)
                click.echo(f"Entropy data saved to: {output}")
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('compressed_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--method', default='simple', type=click.Choice(['simple', 'adaptive']), help='Decompression method')
@click.pass_context
def decompress(ctx, compressed_file, output, method):
    """Decompress text."""
    try:
        model_name = ctx.obj['model']
        compression_model = GPT2TextCompression(model_name=model_name)
        
        click.echo(f"Decompressing file: {compressed_file}")
        
        if method == 'simple':
            with open(compressed_file, 'rb') as f:
                compressed_data = f.read()
            
            decompressed_text = compression_model.decompress_text_simple(compressed_data)
        
        elif method == 'adaptive':
            with open(compressed_file, 'r', encoding='utf-8') as f:
                compressed_data = json.load(f)
            
            decompressed_text = compression_model.decompress_text_adaptive(compressed_data)
        
        click.echo(f"Decompressed length: {len(decompressed_text)} characters")
        
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                f.write(decompressed_text)
            click.echo(f"Decompressed text saved to: {output}")
        else:
            click.echo("Decompressed text:")
            click.echo("-" * 50)
            click.echo(decompressed_text)
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('text', type=str, required=False)
@click.option('--file', '-f', type=click.Path(exists=True), help='Input file path')
@click.pass_context
def analyze(ctx, text, file):
    """Analyze text compressibility and properties."""
    try:
        # Get input text
        if file:
            with open(file, 'r', encoding='utf-8') as f:
                input_text = f.read()
        elif text:
            input_text = text
        else:
            click.echo("Error: Either provide text as argument or use --file option", err=True)
            sys.exit(1)
        
        model_name = ctx.obj['model']
        compression_model = GPT2TextCompression(model_name=model_name)
        completion_model = GPT2TextCompletion(model_name=model_name)
        
        click.echo(f"Analyzing text with model: {model_name}")
        click.echo(f"Text length: {len(input_text)} characters")
        click.echo("-" * 50)
        
        # Get analysis
        analysis = compression_model.analyze_text_compressibility(input_text)
        perplexity = completion_model.get_perplexity(input_text)
        
        click.echo(f"Simple compression ratio: {analysis['simple_compression_ratio']:.2f}")
        click.echo(f"Total entropy: {analysis['total_entropy_bits']:.2f} bits")
        click.echo(f"Average entropy per token: {analysis['average_entropy_per_token']:.2f} bits")
        click.echo(f"Perplexity: {analysis['perplexity']:.2f}")
        click.echo(f"Token count: {analysis['token_count']}")
        click.echo(f"Character count: {analysis['character_count']}")
        click.echo(f"Compressibility score: {analysis['compressibility_score']:.4f}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
def version():
    """Show version information."""
    from . import __version__
    click.echo(f"gpt2tc version {__version__}")


if __name__ == '__main__':
    main()