"""
Data Loading and Management System
Handles CNN/DailyMail dataset loading, preprocessing, and sample management
"""

import json
import os
from typing import Dict, List, Optional, Union
import logging
from pathlib import Path
import pandas as pd

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets library not available. Install with: pip install datasets")

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Professional data loading system for summarization datasets.
    
    Features:
    - CNN/DailyMail dataset loading
    - Sample management and caching
    - Data preprocessing and validation
    - Export/import functionality
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize DataLoader
        
        Args:
            cache_dir: Directory for caching datasets
        """
        self.cache_dir = cache_dir or "./data/cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"DataLoader initialized with cache dir: {self.cache_dir}")
    
    def load_cnn_dailymail(self, 
                          split: str = "test",
                          num_samples: Optional[int] = None,
                          version: str = "3.0.0") -> List[Dict]:
        """
        Load CNN/DailyMail dataset
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            num_samples: Number of samples to load (None for all)
            version: Dataset version
            
        Returns:
            List of dictionaries with 'article' and 'reference_summary' keys
        """
        if not DATASETS_AVAILABLE:
            logger.error("datasets library not available")
            return self._load_sample_data()
        
        logger.info(f"Loading CNN/DailyMail {split} split (version {version})")
        
        try:
            # Load dataset
            dataset = load_dataset('abisee/cnn_dailymail', version, split=split)
            
            # Limit samples if requested
            if num_samples:
                dataset = dataset.select(range(min(num_samples, len(dataset))))
            
            # Convert to our format
            data = []
            for item in dataset:
                data.append({
                    'article': item['article'],
                    'reference_summary': item['highlights'],
                    'id': item.get('id', len(data))
                })
            
            logger.info(f"Loaded {len(data)} samples from CNN/DailyMail")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load CNN/DailyMail: {e}")
            return self._load_sample_data()
    
    def _load_sample_data(self) -> List[Dict]:
        """Load sample data when dataset library is not available"""
        logger.info("Loading built-in sample data")
        
        return [
            {
                'article': """
                Artificial intelligence has revolutionized modern technology in unprecedented ways. 
                Machine learning algorithms enable computers to learn from vast amounts of data without 
                explicit programming. Deep learning neural networks, inspired by the human brain, can 
                now recognize patterns in images, understand natural language, and even generate creative 
                content. Natural language processing has advanced to the point where AI systems can 
                engage in human-like conversations, translate between languages in real-time, and 
                summarize lengthy documents automatically. Computer vision technology allows machines 
                to interpret and understand visual information from the world, powering applications 
                from autonomous vehicles to medical diagnosis systems. The integration of AI across 
                industries has improved efficiency, accuracy, and decision-making capabilities. 
                Healthcare providers use AI to detect diseases earlier and recommend personalized 
                treatments. Financial institutions employ machine learning for fraud detection and 
                algorithmic trading. Manufacturing companies utilize AI-powered robots for precision 
                tasks and quality control. Despite these advances, challenges remain in areas such as 
                algorithmic bias, data privacy, interpretability of AI decisions, and the ethical 
                implications of autonomous systems.
                """,
                'reference_summary': "AI has transformed technology through machine learning, deep learning, and NLP. Applications span healthcare, finance, and manufacturing, though challenges like bias and privacy remain.",
                'id': 1
            },
            {
                'article': """
                Climate change represents one of the most pressing challenges facing humanity in the 
                21st century. Global temperatures have risen significantly over the past century, 
                primarily due to increased greenhouse gas emissions from human activities. The burning 
                of fossil fuels for energy, deforestation, and industrial processes have released 
                enormous amounts of carbon dioxide and methane into the atmosphere. These greenhouse 
                gases trap heat, leading to a warming effect known as the greenhouse effect. The 
                consequences of climate change are already visible worldwide. Polar ice caps and 
                glaciers are melting at alarming rates, contributing to rising sea levels that threaten 
                coastal communities. Extreme weather events, including hurricanes, droughts, floods, 
                and heat waves, have become more frequent and intense. Changes in precipitation patterns 
                affect agriculture and water supplies, potentially leading to food insecurity. Ocean 
                acidification, caused by increased absorption of carbon dioxide, threatens marine 
                ecosystems and the communities that depend on them. Many species face extinction as 
                their habitats change faster than they can adapt.
                """,
                'reference_summary': "Climate change, driven by greenhouse gas emissions, causes rising temperatures, melting ice caps, extreme weather, and threatens ecosystems and human communities worldwide.",
                'id': 2
            },
            {
                'article': """
                Space exploration has captured human imagination for decades and continues to push the 
                boundaries of what's possible. Since the first satellite launch in 1957 and the moon 
                landing in 1969, humanity has made remarkable progress in understanding our universe. 
                Modern space agencies like NASA, ESA, and private companies like SpaceX have developed 
                advanced technologies for space travel. The International Space Station serves as a 
                permanent laboratory orbiting Earth, enabling research in microgravity conditions. 
                Robotic missions have explored nearly every planet in our solar system, sending back 
                invaluable data about planetary geology, atmospheres, and potential for life. Mars has 
                been particularly exciting, with rovers like Curiosity and Perseverance analyzing soil 
                samples and searching for signs of ancient microbial life. Space telescopes such as 
                Hubble and James Webb have revolutionized astronomy, capturing images of distant 
                galaxies and helping scientists understand the universe's origins. Commercial space 
                flight is becoming reality, with companies developing reusable rockets and planning 
                tourist trips to orbit.
                """,
                'reference_summary': "Space exploration has advanced from early satellites to modern missions exploring planets, operating space stations, and developing commercial spaceflight capabilities.",
                'id': 3
            }
        ]
    
    def save_samples(self, data: List[Dict], filename: str) -> bool:
        """
        Save samples to JSON file
        
        Args:
            data: List of sample dictionaries
            filename: Output filename
            
        Returns:
            Success status
        """
        try:
            # Ensure directory exists
            filepath = Path(filename)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(data)} samples to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save samples: {e}")
            return False
    
    def load_samples(self, filename: str) -> List[Dict]:
        """
        Load samples from JSON file
        
        Args:
            filename: Input filename
            
        Returns:
            List of sample dictionaries
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded {len(data)} samples from {filename}")
            return data
            
        except FileNotFoundError:
            logger.warning(f"File not found: {filename}")
            return []
        except Exception as e:
            logger.error(f"Failed to load samples: {e}")
            return []
    
    def validate_data(self, data: List[Dict]) -> Dict:
        """
        Validate dataset structure and content
        
        Args:
            data: List of sample dictionaries
            
        Returns:
            Validation report
        """
        report = {
            'total_samples': len(data),
            'valid_samples': 0,
            'issues': []
        }
        
        required_keys = ['article', 'reference_summary']
        
        for i, sample in enumerate(data):
            # Check required keys
            missing_keys = [key for key in required_keys if key not in sample]
            if missing_keys:
                report['issues'].append(f"Sample {i}: Missing keys {missing_keys}")
                continue
            
            # Check content
            if not sample['article'] or not sample['reference_summary']:
                report['issues'].append(f"Sample {i}: Empty content")
                continue
            
            # Check lengths
            article_words = len(sample['article'].split())
            summary_words = len(sample['reference_summary'].split())
            
            if article_words < 10:
                report['issues'].append(f"Sample {i}: Article too short ({article_words} words)")
                continue
            
            if summary_words < 3:
                report['issues'].append(f"Sample {i}: Summary too short ({summary_words} words)")
                continue
            
            report['valid_samples'] += 1
        
        report['validity_rate'] = report['valid_samples'] / report['total_samples'] if report['total_samples'] > 0 else 0
        
        logger.info(f"Validation: {report['valid_samples']}/{report['total_samples']} valid samples")
        return report
    
    def get_statistics(self, data: List[Dict]) -> Dict:
        """
        Get dataset statistics
        
        Args:
            data: List of sample dictionaries
            
        Returns:
            Statistics dictionary
        """
        if not data:
            return {}
        
        article_lengths = [len(sample['article'].split()) for sample in data]
        summary_lengths = [len(sample['reference_summary'].split()) for sample in data]
        compression_ratios = [s/a for a, s in zip(article_lengths, summary_lengths) if a > 0]
        
        stats = {
            'total_samples': len(data),
            'article_stats': {
                'mean_length': sum(article_lengths) / len(article_lengths),
                'min_length': min(article_lengths),
                'max_length': max(article_lengths),
                'median_length': sorted(article_lengths)[len(article_lengths)//2]
            },
            'summary_stats': {
                'mean_length': sum(summary_lengths) / len(summary_lengths),
                'min_length': min(summary_lengths),
                'max_length': max(summary_lengths),
                'median_length': sorted(summary_lengths)[len(summary_lengths)//2]
            },
            'compression_stats': {
                'mean_ratio': sum(compression_ratios) / len(compression_ratios),
                'min_ratio': min(compression_ratios),
                'max_ratio': max(compression_ratios)
            }
        }
        
        return stats
    
    def export_to_csv(self, data: List[Dict], filename: str) -> bool:
        """
        Export data to CSV format
        
        Args:
            data: List of sample dictionaries
            filename: Output CSV filename
            
        Returns:
            Success status
        """
        try:
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False, encoding='utf-8')
            logger.info(f"Exported {len(data)} samples to {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")
            return False
    
    def create_sample_dataset(self, 
                            full_data: List[Dict], 
                            sample_size: int,
                            strategy: str = "random") -> List[Dict]:
        """
        Create a sample dataset from full data
        
        Args:
            full_data: Complete dataset
            sample_size: Number of samples to select
            strategy: Sampling strategy ('random', 'first', 'balanced')
            
        Returns:
            Sampled dataset
        """
        if sample_size >= len(full_data):
            return full_data
        
        if strategy == "random":
            import random
            return random.sample(full_data, sample_size)
        elif strategy == "first":
            return full_data[:sample_size]
        elif strategy == "balanced":
            # Try to balance by length
            sorted_data = sorted(full_data, key=lambda x: len(x['article'].split()))
            step = len(sorted_data) // sample_size
            return [sorted_data[i * step] for i in range(sample_size)]
        else:
            return full_data[:sample_size]


# Test the DataLoader
if __name__ == "__main__":
    print("=" * 60)
    print("DATA LOADER - PROFESSIONAL TEST")
    print("=" * 60)
    
    # Initialize loader
    loader = DataLoader()
    
    # Load sample data
    data = loader.load_cnn_dailymail(split='test', num_samples=5)
    
    print(f"\nLoaded {len(data)} samples")
    
    # Validate data
    validation = loader.validate_data(data)
    print(f"Validation: {validation['valid_samples']}/{validation['total_samples']} valid")
    
    # Get statistics
    stats = loader.get_statistics(data)
    print(f"\nStatistics:")
    print(f"  Article length: {stats['article_stats']['mean_length']:.1f} words (avg)")
    print(f"  Summary length: {stats['summary_stats']['mean_length']:.1f} words (avg)")
    print(f"  Compression ratio: {stats['compression_stats']['mean_ratio']:.2%}")
    
    # Test save/load
    test_file = "test_samples.json"
    if loader.save_samples(data, test_file):
        loaded_data = loader.load_samples(test_file)
        print(f"\nSave/Load test: {len(loaded_data)} samples loaded")
        
        # Cleanup
        os.remove(test_file)
    
    print("\n" + "=" * 60)