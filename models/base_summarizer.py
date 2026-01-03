"""
Base Summarizer Class
Defines the interface for all summarization models
Implements Strategy Design Pattern for interchangeable algorithms
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseSummarizer(ABC):
    """
    Abstract base class for all summarization models.
    Implements common functionality and defines interface.
    
    Design Pattern: Strategy Pattern
    - Allows switching between different summarization algorithms
    - Ensures consistent interface across models
    """
    
    def __init__(self, model_name: str, model_type: str):
        """
        Initialize base summarizer
        
        Args:
            model_name: Name of the model (e.g., "TextRank", "BART")
            model_type: Type of summarization ("Extractive" or "Abstractive")
        """
        self.model_name = model_name
        self.model_type = model_type
        self.is_initialized = False
        self.stats = {
            'total_summarizations': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0
        }
        logger.info(f"Initializing {model_name} ({model_type}) summarizer")
    
    @abstractmethod
    def summarize(self, text: str, **kwargs) -> str:
        """
        Generate summary from input text.
        Must be implemented by all subclasses.
        
        Args:
            text: Input text to summarize
            **kwargs: Additional parameters specific to each model
            
        Returns:
            Generated summary string
        """
        pass
    
    def summarize_with_metrics(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Summarize text and return detailed metrics
        
        Args:
            text: Input text to summarize
            **kwargs: Model-specific parameters
            
        Returns:
            Dictionary containing summary and metadata
        """
        start_time = time.time()
        
        # Generate summary
        summary = self.summarize(text, **kwargs)
        
        # Calculate metrics
        processing_time = time.time() - start_time
        self._update_stats(processing_time)
        
        return {
            'summary': summary,
            'metadata': {
                'model_name': self.model_name,
                'model_type': self.model_type,
                'processing_time': processing_time,
                'input_length': len(text.split()),
                'summary_length': len(summary.split()),
                'compression_ratio': len(summary.split()) / len(text.split()) if len(text.split()) > 0 else 0,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
    
    def batch_summarize(self, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Summarize multiple texts
        
        Args:
            texts: List of texts to summarize
            **kwargs: Model-specific parameters
            
        Returns:
            List of dictionaries with summaries and metadata
        """
        logger.info(f"Batch summarizing {len(texts)} texts with {self.model_name}")
        results = []
        
        for idx, text in enumerate(texts):
            logger.info(f"Processing text {idx + 1}/{len(texts)}")
            result = self.summarize_with_metrics(text, **kwargs)
            result['metadata']['batch_index'] = idx
            results.append(result)
        
        return results
    
    def _update_stats(self, processing_time: float):
        """Update internal statistics"""
        self.stats['total_summarizations'] += 1
        self.stats['total_processing_time'] += processing_time
        self.stats['average_processing_time'] = (
            self.stats['total_processing_time'] / self.stats['total_summarizations']
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed model information
        
        Returns:
            Dictionary with model specifications
        """
        return {
            'name': self.model_name,
            'type': self.model_type,
            'statistics': self.stats.copy(),
            'is_initialized': self.is_initialized
        }
    
    def reset_stats(self):
        """Reset usage statistics"""
        self.stats = {
            'total_summarizations': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0
        }
        logger.info(f"Statistics reset for {self.model_name}")
    
    def validate_input(self, text: str, min_length: int = 10) -> bool:
        """
        Validate input text
        
        Args:
            text: Input text
            min_length: Minimum number of words required
            
        Returns:
            Boolean indicating if input is valid
            
        Raises:
            ValueError: If input is invalid
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")
        
        word_count = len(text.split())
        if word_count < min_length:
            raise ValueError(
                f"Input text too short. Minimum {min_length} words required, got {word_count}"
            )
        
        return True
    
    def __repr__(self) -> str:
        """String representation of the summarizer"""
        return (f"{self.__class__.__name__}(model_name='{self.model_name}', "
                f"model_type='{self.model_type}', "
                f"total_summarizations={self.stats['total_summarizations']})")


class SummarizerFactory:
    """
    Factory Pattern for creating summarizer instances
    Centralizes model instantiation logic
    """
    
    _models = {}
    
    @classmethod
    def register_model(cls, model_class, name: str):
        """Register a new summarizer model"""
        cls._models[name.lower()] = model_class
        logger.info(f"Registered model: {name}")
    
    @classmethod
    def create_summarizer(cls, model_name: str, **kwargs):
        """
        Create a summarizer instance
        
        Args:
            model_name: Name of the model to create
            **kwargs: Model-specific initialization parameters
            
        Returns:
            Instance of requested summarizer
            
        Raises:
            ValueError: If model not found
        """
        model_name_lower = model_name.lower()
        
        if model_name_lower not in cls._models:
            available = ', '.join(cls._models.keys())
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {available}"
            )
        
        model_class = cls._models[model_name_lower]
        return model_class(**kwargs)
    
    @classmethod
    def list_available_models(cls) -> List[str]:
        """Get list of available models"""
        return list(cls._models.keys())