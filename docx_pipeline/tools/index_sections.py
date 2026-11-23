"""Deprecated: use `python -I tools/docx_cli.py index` / `show <section>` instead."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_cli import main
if __name__ == '__main__':
    sys.argv = [sys.argv[0], 'index']; main()
