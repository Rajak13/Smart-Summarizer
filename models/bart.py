"""
BART (Bidirectional and Auto-Regressive Transformers) Abstractive Summarization
State-of-the-art sequence-to-sequence model for text generation
Professional implementation with comprehensive features
"""

# Handle imports when running directly (python models/bart.py)
# For proper package usage, run as: python -m models.bart
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import logging
from typing import Dict, List, Optional, Union
from models.base_summarizer import BaseSummarizer

logger = logging.getLogger(__name__)


class BARTSummarizer(BaseSummarizer):
    """
    BART implementation for abstractive text summarization.
    
    Model Architecture:
    - Encoder: Bidirectional transformer (like BERT)
    - Decoder: Auto-regressive transformer (like GPT)
    - Pre-trained on denoising tasks
    
    Key Features:
    - Generates human-like, fluent summaries
    - Can paraphrase and compress information
    - Handles long documents effectively
    - State-of-the-art performance on CNN/DailyMail
    
    Training Objective:
    Trained to reconstruct original text from corrupted versions:
    - Token masking
    - Token deletion
    - Sentence permutation
    - Document rotation
    
    Mathematical Foundation:
    Self-Attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
    Where Q=Query, K=Key, V=Value, d_k=dimension of keys
    """
    
    def __init__(self, 
                 model_name: str = "facebook/bart-large-cnn",
                 device: Optional[str] = None,
                 use_fp16: bool = False):
        """
        Initialize BART Summarizer
        
        Args:
            model_name: HuggingFace model identifier
            device: Computing device ('cuda', 'cpu', or None for auto-detect)
            use_fp16: Use 16-bit floating point for faster inference (requires GPU)
        """
        super().__init__(model_name="BART", model_type="Abstractive")
        
        logger.info(f"Loading BART model: {model_name}")
        logger.info("Initial model loading may take 2-3 minutes...")
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        if self.device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Load tokenizer and model
        try:
            self.tokenizer = BartTokenizer.from_pretrained(model_name)
            self.model = BartForConditionalGeneration.from_pretrained(model_name)
            
            # Move model to device
            self.model.to(self.device)
            
            # Enable FP16 if requested and GPU available
            if use_fp16 and self.device == "cuda":
                self.model.half()
                logger.info("Using FP16 precision for faster inference")
            
            # Set to evaluation mode
            self.model.eval()
            
            self.model_name_full = model_name
            self.is_initialized = True
            
            logger.info("BART model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load BART model: {e}")
            raise
    
    def summarize(self,
                  text: str,
                  max_length: int = 150,
                  min_length: int = 50,
                  num_beams: int = 4,
                  length_penalty: float = 2.0,
                  no_repeat_ngram_size: int = 3,
                  early_stopping: bool = True,
                  do_sample: bool = False,
                  temperature: float = 1.0,
                  top_k: int = 50,
                  top_p: float = 0.95) -> str:
        """
        Generate abstractive summary using BART
        
        Beam Search: Maintains top-k hypotheses at each step
        Length Penalty: Exponential penalty applied to sequence length
        
        Args:
            text: Input text to summarize
            max_length: Maximum summary length in tokens
            min_length: Minimum summary length in tokens
            num_beams: Number of beams for beam search (higher = better quality, slower)
            length_penalty: >1.0 favors longer sequences, <1.0 favors shorter
            no_repeat_ngram_size: Prevent repetition of n-grams
            early_stopping: Stop when num_beams hypotheses are complete
            do_sample: Use sampling instead of greedy decoding
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens for sampling
            top_p: Nucleus sampling threshold
            
        Returns:
            Generated summary string
        """
        # Validate input
        self.validate_input(text)
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            max_length=1024,  # BART max input length
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
                # Sampling-based generation (more diverse)
                summary_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    early_stopping=early_stopping
                )
            else:
                # Beam search generation (more deterministic, higher quality)
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
                       batch_size: int = 4,
                       max_length: int = 150,
                       min_length: int = 50,
                       **kwargs) -> List[str]:
        """
        Efficiently summarize multiple texts in batches
        
        Args:
            texts: List of texts to summarize
            batch_size: Number of texts to process simultaneously
            max_length: Maximum summary length
            min_length: Minimum summary length
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated summaries
        """
        logger.info(f"Batch summarizing {len(texts)} texts (batch_size={batch_size})")
        
        summaries = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                max_length=1024,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            # Generate summaries for batch
            with torch.no_grad():
                summary_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=kwargs.get('num_beams', 4),
                    early_stopping=True
                )
            
            # Decode summaries
            batch_summaries = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in summary_ids
            ]
            
            summaries.extend(batch_summaries)
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        return summaries
    
    def get_model_info(self) -> Dict:
        """Return comprehensive model information"""
        info = super().get_model_info()
        info.update({
            'algorithm': 'Transformer Encoder-Decoder',
            'architecture': {
                'encoder': 'Bidirectional (BERT-like)',
                'decoder': 'Auto-regressive (GPT-like)',
                'layers': '12 encoder + 12 decoder',
                'attention_heads': 16,
                'hidden_size': 1024,
                'parameters': '406M'
            },
            'training': {
                'objective': 'Denoising autoencoder',
                'noise_functions': [
                    'Token masking',
                    'Token deletion',
                    'Sentence permutation',
                    'Document rotation'
                ],
                'dataset': 'Large-scale web text + CNN/DailyMail fine-tuning'
            },
            'performance': {
                'rouge_1': '44.16',
                'rouge_2': '21.28',
                'rouge_l': '40.90',
                'benchmark': 'CNN/DailyMail test set'
            },
            'advantages': [
                'Generates fluent, human-like summaries',
                'Can paraphrase and compress effectively',
                'Handles long documents well',
                'State-of-the-art performance'
            ],
            'limitations': [
                'May introduce factual errors',
                'Computationally intensive',
                'Requires GPU for fast inference',
                'Black-box nature (less interpretable)'
            ]
        })
        return info
    
    def __del__(self):
        """Cleanup GPU memory when object is destroyed"""
        if hasattr(self, 'device') and self.device == 'cuda':
            torch.cuda.empty_cache()


# Test the implementation
if __name__ == "__main__":
    sample_text = """
    Machine learning has revolutionized artificial intelligence in recent years. 
    Deep learning neural networks can now perform tasks that were impossible just 
    a decade ago. Computer vision systems can recognize objects in images with 
    superhuman accuracy. Natural language processing models can generate human-like 
    text and translate between languages. Reinforcement learning has enabled AI 
    to master complex games like Go and StarCraft. These advances have been driven 
    by increases in computing power, availability of large datasets, and algorithmic 
    innovations. However, challenges remain in areas like explainability, fairness, 
    and robustness. The field continues to evolve rapidly with new breakthroughs 
    occurring regularly.
    """
    
    print("=" * 70)
    print("BART SUMMARIZER - PROFESSIONAL TEST")
    print("=" * 70)
    
    # Initialize summarizer
    summarizer = BARTSummarizer()
    
    # Generate summary with metrics
    result = summarizer.summarize_with_metrics(
        sample_text,
        max_length=100,
        min_length=30,
        num_beams=4
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
    
    print("\n" + "=" * 70)
    model_info = summarizer.get_model_info()
    print(f"Architecture: {model_info['architecture']}")
    print(f"Performance: {model_info['performance']}")