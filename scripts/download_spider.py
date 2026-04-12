#!/usr/bin/env python3
"""
Download and prepare Spider dataset for SQL Self-Improvement.
Spider: Yale Semantic Parsing and Text-to-SQL Challenge
"""

import os
import json
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

# Spider dataset URL
SPIDER_URL = "https://drive.google.com/uc?export=download&id=1iRDVHLr4mX2wQKSgA9J8Pire73Jahh0m"
SPIDER_TABLES_URL = "https://raw.githubusercontent.com/taoyds/spider/master/tables.json"

DATA_DIR = Path(__file__).parent.parent / "data"


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: str, desc: str = "Downloading"):
    """Download file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_spider():
    """Download Spider dataset."""
    print("=" * 60)
    print("SQL Self-Improvement - Spider Dataset Download")
    print("=" * 60)
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # For Spider, we'll use the HuggingFace version (easier)
    print("\n📥 Downloading Spider from HuggingFace...")
    
    try:
        from datasets import load_dataset
        
        # Load Spider dataset
        spider = load_dataset("spider", trust_remote_code=True)
        
        print(f"\n✅ Dataset loaded!")
        print(f"   Train: {len(spider['train'])} examples")
        print(f"   Validation: {len(spider['validation'])} examples")
        
        # Save to disk
        spider_dir = DATA_DIR / "spider"
        spider_dir.mkdir(exist_ok=True)
        
        # Save as JSON for easier manipulation
        train_data = []
        for item in spider['train']:
            train_data.append({
                'question': item['question'],
                'query': item['query'],
                'db_id': item['db_id'],
            })
        
        val_data = []
        for item in spider['validation']:
            val_data.append({
                'question': item['question'],
                'query': item['query'],
                'db_id': item['db_id'],
            })
        
        with open(spider_dir / "train_spider.json", 'w') as f:
            json.dump(train_data, f, indent=2)
        
        with open(spider_dir / "dev.json", 'w') as f:
            json.dump(val_data, f, indent=2)
        
        print(f"\n📁 Saved to {spider_dir}")
        
        # Also try to get database schemas
        print("\n📥 Downloading database schemas...")
        try:
            tables_path = spider_dir / "tables.json"
            download_file(SPIDER_TABLES_URL, str(tables_path), "tables.json")
            print(f"✅ Schemas saved to {tables_path}")
        except Exception as e:
            print(f"⚠️ Could not download schemas: {e}")
            print("   You may need to download manually from: https://yale-lily.github.io/spider")
        
        # Print sample
        print("\n" + "=" * 60)
        print("📋 Sample from dataset:")
        print("=" * 60)
        sample = train_data[0]
        print(f"Question: {sample['question']}")
        print(f"SQL:      {sample['query']}")
        print(f"DB:       {sample['db_id']}")
        
        # Stats
        print("\n" + "=" * 60)
        print("📊 Dataset Statistics:")
        print("=" * 60)
        print(f"Total examples: {len(train_data) + len(val_data)}")
        print(f"Train: {len(train_data)}")
        print(f"Dev: {len(val_data)}")
        
        # Unique databases
        train_dbs = set(d['db_id'] for d in train_data)
        val_dbs = set(d['db_id'] for d in val_data)
        print(f"Unique databases (train): {len(train_dbs)}")
        print(f"Unique databases (dev): {len(val_dbs)}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nAlternative: Download manually from https://yale-lily.github.io/spider")
        return False


def download_spider_databases():
    """
    Download Spider SQLite databases.
    These are needed for execution-based evaluation.
    """
    print("\n" + "=" * 60)
    print("📥 Spider Databases")
    print("=" * 60)
    print("""
To run execution-based evaluation, you need the SQLite databases.

Download manually from:
https://yale-lily.github.io/spider

Extract to: data/spider/database/

Structure should be:
data/spider/database/
├── academic/
│   └── academic.sqlite
├── activity_1/
│   └── activity_1.sqlite
└── ...
""")


if __name__ == "__main__":
    success = download_spider()
    if success:
        download_spider_databases()
        print("\n✅ Spider dataset ready!")
        print("\nNext steps:")
        print("1. Download SQLite databases manually (see above)")
        print("2. Run: python src/evaluation/eval_dev.py")
    else:
        print("\n❌ Download failed. Please download manually.")
