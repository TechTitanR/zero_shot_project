# verify_setup.py
"""Small script to verify environment and Transformers installation."""
import sys
try:
    import transformers
    import torch
    print('transformers version:', transformers.__version__)
    print('torch version:', torch.__version__)
    from transformers import pipeline
    p = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
    print('Successfully loaded zero-shot pipeline (model downloaded if first run).')
except Exception as e:
    print('Error during verification:', e)
    sys.exit(1)
