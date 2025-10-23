#!/usr/bin/env python3
"""Fix Unicode encoding issues in Jupyter notebooks for Kaggle submission."""

import json
import sys
from pathlib import Path

def clean_unicode(text):
    """Remove or replace problematic Unicode characters."""
    if not isinstance(text, str):
        return text
    
    # Replace surrogates and other problematic characters
    cleaned = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    
    # Additional cleanup for common issues
    cleaned = cleaned.replace('\udcc8', '')  # Remove the specific surrogate mentioned
    cleaned = cleaned.replace('\ufffd', '')  # Remove replacement characters
    
    return cleaned

def fix_notebook(input_path, output_path=None):
    """Fix Unicode issues in a Jupyter notebook."""
    if output_path is None:
        output_path = input_path
    
    print(f"Reading notebook: {input_path}")
    
    # Read with error handling
    with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
        nb = json.load(f)
    
    print(f"Cleaning {len(nb.get('cells', []))} cells...")
    
    # Clean all cells
    for cell in nb.get('cells', []):
        # Clean source
        if 'source' in cell:
            if isinstance(cell['source'], list):
                cell['source'] = [clean_unicode(line) for line in cell['source']]
            else:
                cell['source'] = clean_unicode(cell['source'])
        
        # Clean outputs
        if 'outputs' in cell:
            for output in cell['outputs']:
                if 'text' in output:
                    if isinstance(output['text'], list):
                        output['text'] = [clean_unicode(line) for line in output['text']]
                    else:
                        output['text'] = clean_unicode(output['text'])
                
                if 'data' in output:
                    for key in output['data']:
                        if isinstance(output['data'][key], str):
                            output['data'][key] = clean_unicode(output['data'][key])
                        elif isinstance(output['data'][key], list):
                            output['data'][key] = [clean_unicode(item) for item in output['data'][key]]
    
    # Write cleaned notebook
    print(f"Writing cleaned notebook: {output_path}")
    with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    
    print("✅ Notebook cleaned successfully!")
    return True

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python fix_notebook_encoding.py <notebook.ipynb> [output.ipynb]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file
    
    fix_notebook(input_file, output_file)
