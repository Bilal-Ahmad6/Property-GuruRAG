#!/usr/bin/env python3
"""
Deployment debug script for ChromaDB issues.
This script helps diagnose and fix common ChromaDB problems.
"""

import argparse
import shutil
import sys
from pathlib import Path

# Add parent directory to path to import our local config
sys.path.insert(0, str(Path(__file__).parent))
from config import settings


def check_chromadb_data():
    """Check the ChromaDB data directory for issues."""
    chroma_dir = Path(settings.chroma_persist_dir)
    
    if not chroma_dir.exists():
        print("ChromaDB data directory does not exist.")
        return False
    
    print(f"ChromaDB data directory exists: {chroma_dir}")
    print(f"Contents: {list(chroma_dir.iterdir())}")
    
    # Check for the sqlite database file
    sqlite_file = chroma_dir / "chroma.sqlite3"
    if sqlite_file.exists():
        print(f"SQLite database file exists: {sqlite_file}")
        print(f"File size: {sqlite_file.stat().st_size} bytes")
    else:
        print("SQLite database file does not exist.")
    
    # Check for the collection directory
    collection_dirs = [d for d in chroma_dir.iterdir() if d.is_dir()]
    if collection_dirs:
        print(f"Collection directories found: {collection_dirs}")
        for coll_dir in collection_dirs:
            print(f"Contents of {coll_dir}: {list(coll_dir.iterdir())}")
    else:
        print("No collection directories found.")
    
    return True


def reset_chromadb_data():
    """Reset the ChromaDB data directory."""
    chroma_dir = Path(settings.chroma_persist_dir)
    
    if chroma_dir.exists():
        print(f"Removing existing ChromaDB data directory: {chroma_dir}")
        shutil.rmtree(chroma_dir)
    
    print(f"Creating new ChromaDB data directory: {chroma_dir}")
    chroma_dir.mkdir(parents=True, exist_ok=True)
    
    print("ChromaDB data directory reset complete.")


def main():
    parser = argparse.ArgumentParser(description="Debug ChromaDB issues")
    parser.add_argument("--check", action="store_true", help="Check ChromaDB data directory")
    parser.add_argument("--reset", action="store_true", help="Reset ChromaDB data directory")
    
    args = parser.parse_args()
    
    if args.check:
        check_chromadb_data()
    elif args.reset:
        reset_chromadb_data()
    else:
        print("Please specify an action: --check or --reset")
        return 1
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
