#!/usr/bin/env python
"""
Wrapper script to run training from the project root.
Usage: python train.py --config configs/train.yaml
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adrone.train import main

if __name__ == "__main__":
    main()
