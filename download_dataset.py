#!/usr/bin/env python3
"""
Dataset Downloader for Sentence Importance Classifier
=======================================================

Downloads and prepares real datasets for training.

Supported datasets:
1. CNN/DailyMail - News articles (RECOMMENDED)
2. XSum - BBC articles
3. Multi-News - Multi-document summaries
4. Reddit TIFU - Informal posts
5. BillSum - Legal documents
6. Scientific Papers - PubMed/ArXiv

Requirements:
    pip install datasets

Usage:
    python download_dataset.py --dataset cnn          # Download CNN/DailyMail
    python download_dataset.py --dataset xsum         # Download XSum
    python download_dataset.py --dataset all          # Download all
    python download_dataset.py --list                 # List available datasets
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple
import re

# Check for datasets library
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("⚠️  'datasets' library not found. Install with: pip install datasets")


# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from ml.sentence_classifier import TrainingExample
except ImportError:
    # Standalone mode
    from dataclasses import dataclass
    
    @dataclass
    class TrainingExample:
        sentence: str
        document: str
        position: int
        total_sentences: int
        is_important: int


# Dataset configurations
DATASETS = {
    'cnn': {
        'name': 'CNN/DailyMail',
        'hf_name': 'cnn_dailymail',
        'hf_config': '3.0.0',
        'text_field': 'article',
        'summary_field': 'highlights',
        'size': '~300MB',
        'examples': '300,000+',
        'description': 'News articles with bullet-point highlights',
        'url': 'https://huggingface.co/datasets/cnn_dailymail'
    },
    'xsum': {
        'name': 'XSum (BBC)',
        'hf_name': 'xsum',
        'hf_config': None,
        'text_field': 'document',
        'summary_field': 'summary',
        'size': '~500MB',
        'examples': '227,000',
        'description': 'BBC articles with 1-sentence summaries',
        'url': 'https://huggingface.co/datasets/xsum'
    },
    'multi_news': {
        'name': 'Multi-News',
        'hf_name': 'multi_news',
        'hf_config': None,
        'text_field': 'document',
        'summary_field': 'summary',
        'size': '~250MB',
        'examples': '56,000',
        'description': 'Multi-document news summaries',
        'url': 'https://huggingface.co/datasets/multi_news'
    },
    'reddit': {
        'name': 'Reddit TIFU',
        'hf_name': 'reddit_tifu',
        'hf_config': 'long',
        'text_field': 'documents',
        'summary_field': 'tldr',
        'size': '~120MB',
        'examples': '120,000',
        'description': 'Reddit posts with TL;DR summaries',
        'url': 'https://huggingface.co/datasets/reddit_tifu'
    },
    'billsum': {
        'name': 'BillSum',
        'hf_name': 'billsum',
        'hf_config': None,
        'text_field': 'text',
        'summary_field': 'summary',
        'size': '~100MB',
        'examples': '23,000',
        'description': 'US Congressional bills with summaries',
        'url': 'https://huggingface.co/datasets/billsum'
    },
    'pubmed': {
        'name': 'PubMed Papers',
        'hf_name': 'scientific_papers',
        'hf_config': 'pubmed',
        'text_field': 'article',
        'summary_field': 'abstract',
        'size': '~4.5GB',
        'examples': '300,000',
        'description': 'Scientific papers with abstracts',
        'url': 'https://huggingface.co/datasets/scientific_papers'
    }
}


class DatasetDownloader:
    """Download and prepare datasets for training"""
    
    OUTPUT_DIR = Path.home() / ".cache" / "echonotes" / "datasets"
    
    def __init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    def list_datasets(self):
        """List available datasets"""
        print("\n" + "="*70)
        print("AVAILABLE DATASETS FOR TRAINING")
        print("="*70 + "\n")
        
        for key, ds in DATASETS.items():
            print(f"  [{key}] {ds['name']}")
            print(f"      Size: {ds['size']} | Examples: {ds['examples']}")
            print(f"      {ds['description']}")
            print(f"      URL: {ds['url']}")
            print()
    
    def download(
        self,
        dataset_key: str,
        max_examples: int = 5000,
        split: str = 'train'
    ) -> List[TrainingExample]:
        """
        Download dataset and convert to training examples.
        
        Args:
            dataset_key: Key from DATASETS dict
            max_examples: Maximum examples to download
            split: Dataset split ('train', 'validation', 'test')
        
        Returns:
            List of TrainingExample objects
        """
        if not HAS_DATASETS:
            print("❌ Error: 'datasets' library required")
            print("   Install with: pip install datasets")
            return []
        
        if dataset_key not in DATASETS:
            print(f"❌ Unknown dataset: {dataset_key}")
            print(f"   Available: {', '.join(DATASETS.keys())}")
            return []
        
        config = DATASETS[dataset_key]
        
        print(f"\n{'='*60}")
        print(f"DOWNLOADING: {config['name']}")
        print(f"{'='*60}")
        print(f"Source: {config['url']}")
        print(f"Size: {config['size']}")
        print(f"Max examples: {max_examples}")
        print()
        
        # Download from HuggingFace
        print("⬇️  Downloading from HuggingFace...")
        try:
            if config['hf_config']:
                dataset = load_dataset(
                    config['hf_name'],
                    config['hf_config'],
                    split=split
                )
            else:
                dataset = load_dataset(
                    config['hf_name'],
                    split=split
                )
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return []
        
        print(f"✅ Downloaded {len(dataset)} examples")
        
        # Convert to training examples
        print("\n🔄 Converting to training format...")
        examples = self._convert_dataset(
            dataset,
            config['text_field'],
            config['summary_field'],
            max_examples
        )
        
        print(f"✅ Created {len(examples)} training examples")
        
        # Save to disk
        output_file = self.OUTPUT_DIR / f"{dataset_key}_training.json"
        self._save_examples(examples, output_file)
        print(f"💾 Saved to: {output_file}")
        
        # Stats
        n_important = sum(1 for ex in examples if ex.is_important == 1)
        print(f"\n📊 Class balance:")
        print(f"   Important: {n_important} ({n_important/len(examples)*100:.1f}%)")
        print(f"   Not important: {len(examples) - n_important} ({(len(examples)-n_important)/len(examples)*100:.1f}%)")
        
        return examples
    
    def _convert_dataset(
        self,
        dataset,
        text_field: str,
        summary_field: str,
        max_examples: int
    ) -> List[TrainingExample]:
        """Convert HuggingFace dataset to TrainingExample list"""
        
        examples = []
        count = 0
        total_important = 0
        
        for item in dataset:
            if count >= max_examples:
                break
            
            # Get text and summary
            text = item.get(text_field, '')
            summary = item.get(summary_field, '')
            
            if not text or not summary:
                continue
            
            # Handle list fields (like reddit_tifu)
            if isinstance(text, list):
                text = ' '.join(text)
            
            # Split text into sentences
            sentences = self._split_sentences(text)
            if len(sentences) < 3:
                continue
            
            # Split summary into sentences (for matching)
            summary_sentences = self._split_sentences(summary)
            
            # Get important keywords from summary
            summary_keywords = self._extract_keywords(summary)
            
            # Normalize for matching
            summary_normalized = set()
            for s in summary_sentences:
                summary_normalized.add(self._normalize(s))
            
            # Create examples with better importance detection
            article_important_count = 0
            article_examples = []
            
            for i, sent in enumerate(sentences):
                sent_norm = self._normalize(sent)
                is_important = 0
                
                # Strategy 1: Direct similarity match (lowered threshold)
                for sum_norm in summary_normalized:
                    if self._similarity(sent_norm, sum_norm) > 0.35:
                        is_important = 1
                        break
                
                # Strategy 2: Keyword overlap (if sentence has many summary keywords)
                if is_important == 0:
                    sent_words = set(sent_norm.split())
                    keyword_overlap = len(sent_words & summary_keywords)
                    if keyword_overlap >= 3:  # At least 3 key terms
                        is_important = 1
                
                # Strategy 3: Position-based (first sentence often summarizes)
                if is_important == 0 and i == 0:
                    # First sentence often important in news
                    for sum_norm in summary_normalized:
                        if self._similarity(sent_norm, sum_norm) > 0.25:
                            is_important = 1
                            break
                
                # Strategy 4: Contains numbers/statistics from summary
                if is_important == 0:
                    sent_numbers = set(re.findall(r'\d+(?:\.\d+)?', sent))
                    summary_numbers = set(re.findall(r'\d+(?:\.\d+)?', summary))
                    if sent_numbers and sent_numbers & summary_numbers:
                        is_important = 1
                
                article_examples.append(TrainingExample(
                    sentence=sent,
                    document=text[:2000],
                    position=i,
                    total_sentences=len(sentences),
                    is_important=is_important
                ))
                
                if is_important:
                    article_important_count += 1
            
            # Only add if we found at least 1 important sentence
            if article_important_count > 0:
                examples.extend(article_examples)
                total_important += article_important_count
                count += 1
            
            if count % 500 == 0:
                print(f"   Processed {count} articles... ({total_important} important sentences so far)")
        
        return examples
    
    def _extract_keywords(self, text: str) -> set:
        """Extract important keywords from text (nouns, verbs, numbers)"""
        # Simple keyword extraction - words longer than 4 chars, not stopwords
        stopwords = {
            'the', 'and', 'for', 'that', 'with', 'was', 'were', 'been', 'have',
            'has', 'had', 'this', 'that', 'these', 'those', 'from', 'they',
            'which', 'would', 'could', 'should', 'about', 'into', 'their',
            'there', 'after', 'before', 'more', 'some', 'other', 'also', 'will'
        }
        
        text = text.lower()
        words = re.findall(r'\b[a-z]+\b', text)
        
        keywords = set()
        for w in words:
            if len(w) > 4 and w not in stopwords:
                keywords.add(w)
        
        # Also add numbers
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        keywords.update(numbers)
        
        return keywords
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Clean text
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = re.sub(r'\s+', ' ', text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter
        return [s.strip() for s in sentences if len(s.strip()) > 20]
    
    def _normalize(self, text: str) -> str:
        """Normalize text for matching"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return ' '.join(text.split())
    
    def _similarity(self, s1: str, s2: str) -> float:
        """Word overlap similarity"""
        words1 = set(s1.split())
        words2 = set(s2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _save_examples(self, examples: List[TrainingExample], filepath: Path):
        """Save examples to JSON"""
        data = []
        for ex in examples:
            data.append({
                'sentence': ex.sentence,
                'document': ex.document,
                'position': ex.position,
                'total_sentences': ex.total_sentences,
                'is_important': ex.is_important
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def load_saved(self, dataset_key: str) -> List[TrainingExample]:
        """Load previously downloaded dataset"""
        filepath = self.OUTPUT_DIR / f"{dataset_key}_training.json"
        
        if not filepath.exists():
            print(f"❌ No saved data for '{dataset_key}'")
            print(f"   Run: python download_dataset.py --dataset {dataset_key}")
            return []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        examples = []
        for item in data:
            examples.append(TrainingExample(
                sentence=item['sentence'],
                document=item['document'],
                position=item['position'],
                total_sentences=item['total_sentences'],
                is_important=item['is_important']
            ))
        
        print(f"✅ Loaded {len(examples)} examples from {filepath}")
        return examples


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for Sentence Importance Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download_dataset.py --list                    # List datasets
    python download_dataset.py --dataset cnn             # Download CNN/DailyMail
    python download_dataset.py --dataset xsum --max 1000 # Download 1000 XSum examples
    python download_dataset.py --dataset all             # Download all datasets
    
After downloading, train with:
    python train_model.py --data ~/.cache/echonotes/datasets/cnn_training.json
        """
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available datasets'
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        choices=list(DATASETS.keys()) + ['all'],
        help='Dataset to download'
    )
    parser.add_argument(
        '--max', '-m',
        type=int,
        default=5000,
        help='Maximum examples to download (default: 5000)'
    )
    parser.add_argument(
        '--split', '-s',
        type=str,
        default='train',
        choices=['train', 'validation', 'test'],
        help='Dataset split (default: train)'
    )
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader()
    
    if args.list or not args.dataset:
        downloader.list_datasets()
        print("\nUsage: python download_dataset.py --dataset <name>")
        return
    
    if args.dataset == 'all':
        for key in DATASETS.keys():
            try:
                downloader.download(key, args.max, args.split)
            except Exception as e:
                print(f"⚠️  Failed to download {key}: {e}")
    else:
        downloader.download(args.dataset, args.max, args.split)
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("""
1. Train the model:
   python train_model.py --data ~/.cache/echonotes/datasets/<dataset>_training.json

2. Or combine multiple datasets:
   python train_model.py --data ~/.cache/echonotes/datasets/cnn_training.json

3. Test the model:
   python -c "from ml import SentenceImportanceClassifier; c = SentenceImportanceClassifier(); c.load_model(); print(c.predict('This is important.', 'Test document.'))"
    """)


if __name__ == "__main__":
    main()
