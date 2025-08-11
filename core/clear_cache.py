import shutil
import os
from pathlib import Path


# Path to the problematic model cache
cache_path = Path.home() / ".cache" / "huggingface" / "hub" / "models--Vamsi--T5_Paraphrase_Paws"

if cache_path.exists():
    shutil.rmtree(cache_path)
    print("Cache cleared. Model will re-download on next use.")
else:
    print("Cache folder not found - it may already be cleared.")
    