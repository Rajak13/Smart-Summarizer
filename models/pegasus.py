"""
PEGASUS (Pre-training with Extracted Gap-sentences for Abstractive SUmmarization)
State-of-the-art model specifically designed for summarization tasks
Professional implementation with Gap Sentence Generation pre-training
"""

# Handle imports when running directly (python models/pegasus.py)
# For proper package usage, run as: python -m models.pegasus
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
import logging
from typing import Dict, List, Optional
from models.base_summarizer import BaseSummarizer

logger = logging.getLogger(__name__)


class PEGASUSSummarizer(BaseSummarizer):
    """
    PEGASUS implementation for abstractive text summarization.
    
    Innovation: Gap Sentence Generation (GSG)
    - Pre-training task: Predict important missing sentences
    - Directly aligned with summarization objective
    - Superior transfer learning for summarization
    
    Model Architecture:
    - Transformer encoder-decoder (16 layers each)
    - Pre-trained on C4 and HugeNews datasets
    - Fine-tuned on domain-specific summarization data
    
    Key Advantages:
    - Highest ROUGE scores on multiple benchmarks
    - Excellent zero-shot and few-shot capabilities
    - Generates highly coherent summaries
    - Handles long documents effectively
    
    Performance Highlights (CNN/DailyMail):
    - ROUGE-1: 44.17
    - ROUGE-2: 21.47
    - ROUGE-L: 41.11
    
    Mathematical Foundation:
    Sentence Importance: ROUGE-F1(Si, D\Si)
    Where Si = sentence i, D\Si = document without sentence i
    """
    
    def __init__(self,
                 model_name: str = "google/pegasus-cnn_dailymail",
                 device: Optional[str] = None,
                 use_fp16: bool = False):
        """
        Initialize PEGASUS Summarizer
        
        Args:
            model_name: HuggingFace model identifier
                       Options: 'google/pegasus-cnn_dailymail' (recommended)
                               'google/pegasus-xsum' (for extreme summarization)
                               'google/pegasus-large' (base model)
            device: Computing device ('cuda', 'cpu', or None for auto-detect)
            use_fp16: Use 16-bit floating point for faster inference
        """
        super().__init__(model_name="PEGASUS", model_type="Abstractive")
        
        logger.info(f"Loading PEGASUS model: {model_name}")
        logger.info("PEGASUS is a large model. Initial loading may take 3-5 minutes...")
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        try:
            logger.info("Loading tokenizer...")
            self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
            
            logger.info("Loading model weights...")
            self.model = PegasusForConditionalGeneration.from_pretrained(model_name)
            
            # Move to device
            self.model.to(self.device)
            
            # Enable FP16 if requested
            if use_fp16 and self.device == "cuda":
                self.model.half()
                logger.info("Using FP16 precision")
            
            # Set to evaluation mode
            self.model.eval()
            
            self.model_name_full = model_name
            self.is_initialized = True
            
            # Get model configuration
            self.config = self.model.config
            
            logger.info("PEGASUS model loaded successfully!")
            logger.info(f"Model size: {self._count_parameters() / 1e6:.1f}M parameters")
            
        except Exception as e:
            logger.error(f"Failed to load PEGASUS model: {e}")
            raise
    
    def _count_parameters(self) -> int:
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def summarize(self,
                  text: str,
                  max_length: int = 128,
                  min_length: int = 32,
                  num_beams: int = 4,
                  length_penalty: float = 2.0,
                  no_repeat_ngram_size: int = 3,
                  early_stopping: bool = True,
                  do_sample: bool = False,
                  temperature: float = 1.0) -> str:
        """
        Generate abstractive summary using PEGASUS
        
        PEGASUS uses special tokens:
        - <pad>: Padding token (also used as decoder start token)
        - </s>: End of sequence token
        - <unk>: Unknown token
        - <mask_1>, <mask_2>: Gap sentence masks
        
        Args:
            text: Input text to summarize
            max_length: Maximum summary length in tokens (PEGASUS optimal: 128)
            min_length: Minimum summary length in tokens
            num_beams: Beam search width (4-8 recommended)
            length_penalty: Controls summary length (>1.0 = longer)
            no_repeat_ngram_size: Prevent n-gram repetition
            early_stopping: Stop when beams complete
            do_sample: Use sampling instead of beam search
            temperature: Sampling randomness (lower = more deterministic)
            
        Returns:
            Generated summary string
        """
        # Validate input
        self.validate_input(text)
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            max_length=1024,  # PEGASUS max input
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Generate summary
        with torch.no_grad():
            if do_sample:
                # Sampling-based generation
                summary_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=True,
                    temperature=temperature,
                    top_k=50,
                    top_p=0.95,
                    no_repeat_ngram_size=no_repeat_ngram_size
                )
            else:
                # Beam search generation (recommended for PEGASUS)
                summary_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    early_stopping=early_stopping
                )
        
        # Decode summary
        summary = self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return summary
    
    def batch_summarize(self,
                       texts: List[str],
                       batch_size: int = 2,
                       max_length: int = 128,
                       **kwargs) -> List[str]:
        """
        Batch summarization (PEGASUS is large, use smaller batches)
        
        Args:
            texts: List of texts to summarize
            batch_size: Texts per batch (2-4 recommended for PEGASUS)
            max_length: Maximum summary length
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated summaries
        """
        logger.info(f"Batch summarizing {len(texts)} texts (batch_size={batch_size})")
        
        summaries = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                max_length=1024,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            # Generate
            with torch.no_grad():
                summary_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    num_beams=kwargs.get('num_beams', 4),
                    length_penalty=kwargs.get('length_penalty', 2.0),
                    early_stopping=True
                )
            
            # Decode
            batch_summaries = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in summary_ids
            ]
            
            summaries.extend(batch_summaries)
            
            logger.info(f"Completed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        return summaries
    
    def get_model_info(self) -> Dict:
        """Return comprehensive model information"""
        info = super().get_model_info()
        info.update({
            'algorithm': 'Gap Sentence Generation (GSG) + Transformer',
            'innovation': 'Pre-training specifically designed for summarization',
            'architecture': {
                'encoder_layers': 16,
                'decoder_layers': 16,
                'attention_heads': 16,
                'hidden_size': 1024,
                'parameters': f'{self._count_parameters() / 1e6:.1f}M',
                'vocabulary_size': self.tokenizer.vocab_size
            },
            'pre_training': {
                'objective': 'Gap Sentence Generation (GSG)',
                'method': 'Mask and predict important sentences',
                'datasets': ['C4 corpus', 'HugeNews dataset'],
                'sentence_selection': 'ROUGE-based importance scoring'
            },
            'fine_tuning': {
                'dataset': 'CNN/DailyMail',
                'task': 'Abstractive summarization'
            },
            'performance': {
                'rouge_1': '44.17',
                'rouge_2': '21.47',
                'rouge_l': '41.11',
                'benchmark': 'CNN/DailyMail test set',
                'ranking': 'State-of-the-art (as of 2020)'
            },
            'advantages': [
                'Highest ROUGE scores on benchmarks',
                'Excellent zero-shot performance',
                'Generates highly coherent summaries',
                'Pre-training aligned with summarization',
                'Strong transfer learning capabilities'
            ],
            'limitations': [
                'Very large model (high memory requirements)',
                'Slower inference than smaller models',
                'May hallucinate facts',
                'Less interpretable (black-box)',
                'Requires powerful GPU for real-time use'
            ],
            'optimal_use_cases': [
                'High-quality abstractive summaries needed',
                'News article summarization',
                'Long document summarization',
                'Multi-document summarization',
                'Research paper abstracts'
            ]
        })
        return info
    
    def get_special_tokens(self) -> Dict:
        """Get information about special tokens"""
        return {
            'pad_token': self.tokenizer.pad_token,
            'eos_token': self.tokenizer.eos_token,
            'unk_token': self.tokenizer.unk_token,
            'mask_token': self.tokenizer.mask_token,
            'vocab_size': self.tokenizer.vocab_size
        }
    
    def __del__(self):
        """Cleanup GPU memory"""
        if hasattr(self, 'device') and self.device == 'cuda':
            torch.cuda.empty_cache()
            logger.info("Cleared GPU cache")


# Test the implementation
if __name__ == "__main__":
    sample_text = """
    Climate change poses one of the greatest challenges to humanity in the 21st century.
    Rising global temperatures are causing ice caps to melt and sea levels to rise.
    Extreme weather events like hurricanes, droughts, and floods are becoming more frequent.
    Scientists warn that without immediate action, the consequences could be catastrophic.
    Renewable energy sources like solar and wind power offer sustainable alternatives to
    fossil fuels. Many countries have committed to reducing carbon emissions through the
    Paris Agreement. However, implementing these changes requires unprecedented international
    cooperation and technological innovation. The transition to a green economy will create
    new jobs while protecting the environment for future generations.
    """
    
    print("=" * 70)
    print("PEGASUS SUMMARIZER - PROFESSIONAL TEST")
    print("=" * 70)
    
    # Initialize summarizer
    summarizer = PEGASUSSummarizer()
    
    # Generate summary with metrics
    result = summarizer.summarize_with_metrics(
        sample_text,
        max_length=100,
        min_length=30,
        num_beams=4,
        length_penalty=2.0
    )
    
    print(f"\nModel: {result['metadata']['model_name']}")
    print(f"Type: {result['metadata']['model_type']}")
    print(f"Device: {summarizer.device}")
    print(f"Input Length: {result['metadata']['input_length']} words")
    print(f"Summary Length: {result['metadata']['summary_length']} words")
    print(f"Compression Ratio: {result['metadata']['compression_ratio']:.2%}")
    print(f"Processing Time: {result['metadata']['processing_time']:.4f} seconds")
    
    print(f"\n{'Generated Summary:':-^70}")
    print(result['summary'])
    
    print(f"\n{'Model Architecture:':-^70}")
    model_info = summarizer.get_model_info()
    print(f"Parameters: {model_info['architecture']['parameters']}")
    print(f"Pre-training: {model_info['pre_training']['objective']}")
    print(f"Performance (CNN/DM): ROUGE-1={model_info['performance']['rouge_1']}, "
          f"ROUGE-2={model_info['performance']['rouge_2']}, "
          f"ROUGE-L={model_info['performance']['rouge_l']}")
    
    print("\n" + "=" * 70)