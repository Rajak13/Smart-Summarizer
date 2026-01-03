"""
TextRank Extractive Summarization
Graph-based ranking algorithm inspired by PageRank
Professional implementation with extensive documentation
"""

# Handle imports when running directly (python models/textrank.py)
# For proper package usage, run as: python -m models.textrank
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import networkx as nx
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import Dict, List, Tuple, Optional
from models.base_summarizer import BaseSummarizer

# Setup logging
logger = logging.getLogger(__name__)


class TextRankSummarizer(BaseSummarizer):
    """
    TextRank implementation for extractive text summarization.
    
    Algorithm Overview:
    1. Split text into sentences
    2. Create TF-IDF vectors for each sentence
    3. Calculate cosine similarity between all sentence pairs
    4. Build weighted graph (sentences as nodes, similarities as edges)
    5. Apply PageRank algorithm to rank sentences
    6. Select top-ranked sentences for summary
    
    Advantages:
    - Fast and efficient (no neural networks)
    - Language-agnostic (works on any language)
    - Interpretable results
    - No training required
    
    Limitations:
    - Cannot generate new sentences
    - May select redundant information
    - Limited semantic understanding
    """
    
    def __init__(self, 
                 damping: float = 0.85,
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 summary_ratio: float = 0.3,
                 min_sentence_length: int = 5):
        """
        Initialize TextRank Summarizer
        
        Args:
            damping: PageRank damping factor (0-1). Higher = more weight to neighbors
            max_iter: Maximum iterations for PageRank convergence
            tol: Convergence tolerance for PageRank
            summary_ratio: Proportion of sentences to include (0-1)
            min_sentence_length: Minimum words per sentence to consider
        """
        super().__init__(model_name="TextRank", model_type="Extractive")
        
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol
        self.summary_ratio = summary_ratio
        self.min_sentence_length = min_sentence_length
        
        # Initialize stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            logger.warning("NLTK stopwords not found. Downloading...")
            import nltk
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
        
        self.is_initialized = True
        logger.info("TextRank summarizer initialized successfully")
    
    def preprocess(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Preprocess text into sentences
        
        Args:
            text: Input text string
            
        Returns:
            Tuple of (original_sentences, cleaned_sentences)
        """
        # Split into sentences
        sentences = sent_tokenize(text)
        
        # Filter out very short sentences
        filtered_sentences = [
            s for s in sentences 
            if len(s.split()) >= self.min_sentence_length
        ]
        
        if not filtered_sentences:
            filtered_sentences = sentences  # Keep all if filtering removes everything
        
        # Clean sentences for similarity calculation
        cleaned_sentences = []
        for sent in filtered_sentences:
            # Tokenize and lowercase
            words = word_tokenize(sent.lower())
            # Remove stopwords and non-alphanumeric tokens
            words = [w for w in words if w.isalnum() and w not in self.stop_words]
            cleaned_sentences.append(' '.join(words))
        
        return filtered_sentences, cleaned_sentences
    
    def build_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """
        Build sentence similarity matrix using TF-IDF and cosine similarity
        
        Mathematical Foundation:
        - TF-IDF: Term Frequency-Inverse Document Frequency
        - Cosine Similarity: cos(θ) = (A·B) / (||A|| × ||B||)
        
        Args:
            sentences: List of cleaned sentences
            
        Returns:
            Similarity matrix (numpy array) of shape [n_sentences, n_sentences]
        """
        # Edge case handling
        n_sentences = len(sentences)
        if n_sentences < 2:
            return np.zeros((n_sentences, n_sentences))
        
        # Remove empty sentences
        valid_sentences = [s for s in sentences if s.strip()]
        if not valid_sentences:
            return np.zeros((n_sentences, n_sentences))
        
        try:
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(
                max_features=1000,  # Limit features for efficiency
                ngram_range=(1, 2)  # Use unigrams and bigrams
            )
            tfidf_matrix = vectorizer.fit_transform(valid_sentences)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Set diagonal to 0 (sentence shouldn't be similar to itself)
            np.fill_diagonal(similarity_matrix, 0)
            
            return similarity_matrix
            
        except ValueError as e:
            logger.error(f"Error building similarity matrix: {e}")
            return np.zeros((n_sentences, n_sentences))
    
    def calculate_pagerank(self, similarity_matrix: np.ndarray) -> Dict[int, float]:
        """
        Apply PageRank algorithm to rank sentences
        
        PageRank Formula:
        WS(Vi) = (1-d) + d × Σ(wji / Σwjk) × WS(Vj)
        
        Where:
        - WS(Vi) = Score of sentence i
        - d = damping factor
        - wji = weight of edge from sentence j to i
        
        Args:
            similarity_matrix: Sentence similarity matrix
            
        Returns:
            Dictionary mapping sentence index to score
        """
        # Create graph from similarity matrix
        nx_graph = nx.from_numpy_array(similarity_matrix)
        
        try:
            # Calculate PageRank scores
            scores = nx.pagerank(
                nx_graph,
                alpha=self.damping,  # damping factor
                max_iter=self.max_iter,
                tol=self.tol
            )
            return scores
            
        except Exception as e:
            logger.error(f"PageRank calculation failed: {e}")
            # Return uniform scores as fallback
            n_nodes = similarity_matrix.shape[0]
            return {i: 1.0/n_nodes for i in range(n_nodes)}
    
    def summarize(self, 
                  text: str, 
                  num_sentences: Optional[int] = None,
                  return_scores: bool = False) -> str:
        """
        Generate extractive summary using TextRank
        
        Args:
            text: Input text to summarize
            num_sentences: Number of sentences in summary (overrides ratio)
            return_scores: If True, return tuple of (summary, scores)
            
        Returns:
            Summary string, or tuple of (summary, scores) if return_scores=True
        """
        # Validate input
        self.validate_input(text)
        
        # Preprocess
        original_sentences, cleaned_sentences = self.preprocess(text)
        
        # Edge cases
        if len(original_sentences) == 0:
            return "" if not return_scores else ("", {})
        if len(original_sentences) == 1:
            summary = original_sentences[0]
            return summary if not return_scores else (summary, {0: 1.0})
        
        # Build similarity matrix
        similarity_matrix = self.build_similarity_matrix(cleaned_sentences)
        
        # Calculate sentence scores using PageRank
        scores = self.calculate_pagerank(similarity_matrix)
        
        # Determine number of sentences for summary
        if num_sentences is None:
            num_sentences = max(1, int(len(original_sentences) * self.summary_ratio))
        num_sentences = min(num_sentences, len(original_sentences))
        
        # Rank sentences by score
        ranked_sentences = sorted(
            ((scores[i], i, s) for i, s in enumerate(original_sentences)),
            reverse=True
        )
        
        # Select top sentences and maintain original order
        top_sentences = sorted(
            ranked_sentences[:num_sentences],
            key=lambda x: x[1]  # Sort by original position
        )
        
        # Build summary
        summary = ' '.join([sent for _, _, sent in top_sentences])
        
        if return_scores:
            return summary, {
                'sentence_scores': scores,
                'selected_indices': [idx for _, idx, _ in top_sentences],
                'num_sentences_original': len(original_sentences),
                'num_sentences_summary': num_sentences
            }
        
        return summary
    
    def get_sentence_importance(self, text: str) -> List[Tuple[str, float]]:
        """
        Get all sentences with their importance scores
        
        Args:
            text: Input text
            
        Returns:
            List of (sentence, score) tuples sorted by importance
        """
        original_sentences, cleaned_sentences = self.preprocess(text)
        
        if len(original_sentences) < 2:
            return [(s, 1.0) for s in original_sentences]
        
        similarity_matrix = self.build_similarity_matrix(cleaned_sentences)
        scores = self.calculate_pagerank(similarity_matrix)
        
        # Combine sentences with scores
        sentence_importance = [
            (original_sentences[i], scores[i])
            for i in range(len(original_sentences))
        ]
        
        # Sort by importance
        sentence_importance.sort(key=lambda x: x[1], reverse=True)
        
        return sentence_importance
    
    def get_model_info(self) -> Dict:
        """Return detailed model information"""
        info = super().get_model_info()
        info.update({
            'algorithm': 'Graph-based PageRank',
            'parameters': {
                'damping_factor': self.damping,
                'max_iterations': self.max_iter,
                'tolerance': self.tol,
                'summary_ratio': self.summary_ratio,
                'min_sentence_length': self.min_sentence_length
            },
            'complexity': 'O(V²) where V = number of sentences',
            'advantages': [
                'Fast and efficient',
                'No training required',
                'Language-agnostic',
                'Interpretable results'
            ],
            'limitations': [
                'Cannot generate new sentences',
                'Limited semantic understanding',
                'May miss context'
            ]
        })
        return info


# Test the implementation
if __name__ == "__main__":
    # Sample academic text
    sample_text = """
    Artificial intelligence has become one of the most transformative technologies 
    of the 21st century. Machine learning, a subset of AI, enables computers to 
    learn from data without explicit programming. Deep learning uses neural networks 
    with multiple layers to process complex patterns. Natural language processing 
    allows machines to understand and generate human language. Computer vision enables 
    machines to interpret visual information from the world. AI applications span 
    healthcare, finance, education, transportation, and entertainment. Ethical 
    considerations around AI include privacy, bias, and job displacement. The future 
    of AI promises both unprecedented opportunities and significant challenges that 
    society must navigate carefully.
    """
    
    # Initialize summarizer
    summarizer = TextRankSummarizer(summary_ratio=0.3)
    
    print("=" * 70)
    print("TEXTRANK SUMMARIZER - PROFESSIONAL TEST")
    print("=" * 70)
    
    # Generate summary with metrics
    result = summarizer.summarize_with_metrics(sample_text)
    
    print(f"\nModel: {result['metadata']['model_name']}")
    print(f"Type: {result['metadata']['model_type']}")
    print(f"Input Length: {result['metadata']['input_length']} words")
    print(f"Summary Length: {result['metadata']['summary_length']} words")
    print(f"Compression Ratio: {result['metadata']['compression_ratio']:.2%}")
    print(f"Processing Time: {result['metadata']['processing_time']:.4f} seconds")
    
    print(f"\n{'Summary:':-^70}")
    print(result['summary'])
    
    print(f"\n{'Sentence Importance Ranking:':-^70}")
    importance = summarizer.get_sentence_importance(sample_text)
    for i, (sent, score) in enumerate(importance[:5], 1):
        print(f"{i}. [Score: {score:.4f}] {sent[:80]}...")
    
    print("\n" + "=" * 70)
    print(summarizer.get_model_info())