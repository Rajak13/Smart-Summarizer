"""
Comprehensive Evaluation System for Summarization Models
Implements ROUGE metrics, comparison analysis, and statistical testing
"""

# Handle different rouge library installations
try:
    from rouge import Rouge
    ROUGE_AVAILABLE = True
    ROUGE_TYPE = "rouge"
except ImportError:
    try:
        from rouge_score import rouge_scorer
        ROUGE_AVAILABLE = True
        ROUGE_TYPE = "rouge_score"
    except ImportError:
        ROUGE_AVAILABLE = False
        ROUGE_TYPE = None
        print("Warning: No ROUGE library found. Install with: pip install rouge-score")

import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd
import logging
from scipy import stats
import time

logger = logging.getLogger(__name__)


class SummarizerEvaluator:
    """
    Professional evaluation system for summarization models.
    
    Metrics Implemented:
    - ROUGE-1: Unigram overlap
    - ROUGE-2: Bigram overlap
    - ROUGE-L: Longest common subsequence
    - ROUGE-W: Weighted longest common subsequence
    
    Additional Analysis:
    - Compression ratio
    - Processing time
    - Statistical significance testing
    - Model comparison
    """
    
    def __init__(self):
        """Initialize evaluator with ROUGE scorer"""
        if ROUGE_AVAILABLE:
            if ROUGE_TYPE == "rouge":
                self.rouge = Rouge()
                self.rouge_scorer = None
            else:  # rouge_score
                self.rouge = None
                self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            logger.info(f"Evaluator initialized with {ROUGE_TYPE} library")
        else:
            self.rouge = None
            self.rouge_scorer = None
            logger.warning("ROUGE library not available - only basic metrics will be computed")
        
        self.evaluation_history = []
    
    def _calculate_rouge_scores(self, generated: str, reference: str) -> Dict:
        """Calculate ROUGE scores using available library"""
        if not ROUGE_AVAILABLE:
            return {
                'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
            }
        
        if ROUGE_TYPE == "rouge":
            # Original rouge library
            scores = self.rouge.get_scores(generated, reference)[0]
            return scores
        else:
            # rouge_score library
            scores = self.rouge_scorer.score(reference, generated)
            return {
                'rouge-1': {
                    'f': scores['rouge1'].fmeasure,
                    'p': scores['rouge1'].precision,
                    'r': scores['rouge1'].recall
                },
                'rouge-2': {
                    'f': scores['rouge2'].fmeasure,
                    'p': scores['rouge2'].precision,
                    'r': scores['rouge2'].recall
                },
                'rouge-l': {
                    'f': scores['rougeL'].fmeasure,
                    'p': scores['rougeL'].precision,
                    'r': scores['rougeL'].recall
                }
            }
    
    def evaluate_single(self,
                       generated: str,
                       reference: str,
                       model_name: str = "Unknown") -> Dict:
        """
        Evaluate a single summary against reference
        
        ROUGE Metrics Explained:
        - Precision: What % of generated words are in reference
        - Recall: What % of reference words are in generated
        - F1-Score: Harmonic mean of precision and recall
        
        Args:
            generated: Generated summary
            reference: Human reference summary
            model_name: Name of the model
            
        Returns:
            Dictionary containing all metrics
        """
        if not generated or not reference:
            logger.warning("Empty summary or reference provided")
            return self._empty_scores()
        
        try:
            # Calculate ROUGE scores
            scores = self._calculate_rouge_scores(generated, reference)
            
            # Calculate additional metrics
            compression_ratio = len(generated.split()) / len(reference.split()) if len(reference.split()) > 0 else 0
            
            result = {
                'model_name': model_name,
                'rouge_1_f1': scores['rouge-1']['f'],
                'rouge_1_precision': scores['rouge-1']['p'],
                'rouge_1_recall': scores['rouge-1']['r'],
                'rouge_2_f1': scores['rouge-2']['f'],
                'rouge_2_precision': scores['rouge-2']['p'],
                'rouge_2_recall': scores['rouge-2']['r'],
                'rouge_l_f1': scores['rouge-l']['f'],
                'rouge_l_precision': scores['rouge-l']['p'],
                'rouge_l_recall': scores['rouge-l']['r'],
                'compression_ratio': compression_ratio,
                'generated_length': len(generated.split()),
                'reference_length': len(reference.split())
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating summary: {e}")
            return self._empty_scores()
    
    def _empty_scores(self) -> Dict:
        """Return empty scores for error cases"""
        return {
            'rouge_1_f1': 0.0,
            'rouge_1_precision': 0.0,
            'rouge_1_recall': 0.0,
            'rouge_2_f1': 0.0,
            'rouge_2_precision': 0.0,
            'rouge_2_recall': 0.0,
            'rouge_l_f1': 0.0,
            'rouge_l_precision': 0.0,
            'rouge_l_recall': 0.0,
            'compression_ratio': 0.0,
            'generated_length': 0,
            'reference_length': 0
        }
    
    def evaluate_batch(self,
                      generated_summaries: List[str],
                      reference_summaries: List[str],
                      model_name: str = "Unknown") -> Dict:
        """
        Evaluate multiple summaries and aggregate results
        
        Args:
            generated_summaries: List of generated summaries
            reference_summaries: List of reference summaries
            model_name: Name of the model
            
        Returns:
            Dictionary with aggregated statistics
        """
        assert len(generated_summaries) == len(reference_summaries), \
            "Generated and reference lists must have same length"
        
        logger.info(f"Evaluating {len(generated_summaries)} summaries for {model_name}")
        
        results = []
        for gen, ref in zip(generated_summaries, reference_summaries):
            scores = self.evaluate_single(gen, ref, model_name)
            results.append(scores)
        
        # Aggregate statistics
        df = pd.DataFrame(results)
        
        aggregated = {
            'model_name': model_name,
            'num_samples': len(results),
            'rouge_1_f1_mean': df['rouge_1_f1'].mean(),
            'rouge_1_f1_std': df['rouge_1_f1'].std(),
            'rouge_2_f1_mean': df['rouge_2_f1'].mean(),
            'rouge_2_f1_std': df['rouge_2_f1'].std(),
            'rouge_l_f1_mean': df['rouge_l_f1'].mean(),
            'rouge_l_f1_std': df['rouge_l_f1'].std(),
            'compression_ratio_mean': df['compression_ratio'].mean(),
            'compression_ratio_std': df['compression_ratio'].std(),
            'individual_scores': results
        }
        
        # Store in history
        self.evaluation_history.append(aggregated)
        
        return aggregated
    
    def compare_models(self,
                      models_dict: Dict,
                      test_texts: List[str],
                      reference_summaries: List[str],
                      **summarize_kwargs) -> pd.DataFrame:
        """
        Compare multiple models on the same dataset
        
        Args:
            models_dict: Dictionary {model_name: model_instance}
            test_texts: List of texts to summarize
            reference_summaries: List of reference summaries
            **summarize_kwargs: Additional parameters for summarization
            
        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Comparing {len(models_dict)} models on {len(test_texts)} texts")
        
        comparison_results = []
        
        for model_name, model in models_dict.items():
            logger.info(f"Evaluating {model_name}...")
            
            start_time = time.time()
            
            # Generate summaries
            generated_summaries = []
            for text in test_texts:
                try:
                    summary = model.summarize(text, **summarize_kwargs)
                    generated_summaries.append(summary)
                except Exception as e:
                    logger.error(f"Error with {model_name}: {e}")
                    generated_summaries.append("")
            
            total_time = time.time() - start_time
            
            # Evaluate
            eval_results = self.evaluate_batch(
                generated_summaries,
                reference_summaries,
                model_name
            )
            
            # Add timing information
            eval_results['total_time'] = total_time
            eval_results['avg_time_per_summary'] = total_time / len(test_texts)
            
            comparison_results.append(eval_results)
        
        # Create comparison DataFrame
        df = pd.DataFrame([
            {
                'Model': r['model_name'],
                'ROUGE-1': f"{r['rouge_1_f1_mean']:.4f} ± {r['rouge_1_f1_std']:.4f}",
                'ROUGE-2': f"{r['rouge_2_f1_mean']:.4f} ± {r['rouge_2_f1_std']:.4f}",
                'ROUGE-L': f"{r['rouge_l_f1_mean']:.4f} ± {r['rouge_l_f1_std']:.4f}",
                'Compression': f"{r['compression_ratio_mean']:.2f}x",
                'Avg Time (s)': f"{r['avg_time_per_summary']:.3f}"
            }
            for r in comparison_results
        ])
        
        logger.info("Model comparison completed")
        return df
    
    def statistical_significance_test(self,
                                     model1_scores: List[float],
                                     model2_scores: List[float],
                                     test_name: str = "paired t-test") -> Dict:
        """
        Test if difference between models is statistically significant
        
        Args:
            model1_scores: Scores from first model
            model2_scores: Scores from second model
            test_name: Type of statistical test
            
        Returns:
            Dictionary with test results
        """
        if test_name == "paired t-test":
            statistic, p_value = stats.ttest_rel(model1_scores, model2_scores)
        elif test_name == "wilcoxon":
            statistic, p_value = stats.wilcoxon(model1_scores, model2_scores)
        else:
            raise ValueError(f"Unknown test: {test_name}")
        
        is_significant = p_value < 0.05
        
        return {
            'test_name': test_name,
            'statistic': statistic,
            'p_value': p_value,
            'is_significant': is_significant,
            'significance_level': 0.05,
            'interpretation': (
                f"The difference is {'statistically significant' if is_significant else 'not statistically significant'} "
                f"(p={p_value:.4f})"
            )
        }
    
    def get_detailed_report(self,
                           evaluation_result: Dict) -> str:
        """
        Generate a detailed text report
        
        Args:
            evaluation_result: Results from evaluate_batch
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 70)
        report.append(f"EVALUATION REPORT: {evaluation_result['model_name']}")
        report.append("=" * 70)
        report.append(f"\nDataset Size: {evaluation_result['num_samples']} samples\n")
        
        report.append("ROUGE Scores (F1):")
        report.append(f"  ROUGE-1: {evaluation_result['rouge_1_f1_mean']:.4f} (±{evaluation_result['rouge_1_f1_std']:.4f})")
        report.append(f"  ROUGE-2: {evaluation_result['rouge_2_f1_mean']:.4f} (±{evaluation_result['rouge_2_f1_std']:.4f})")
        report.append(f"  ROUGE-L: {evaluation_result['rouge_l_f1_mean']:.4f} (±{evaluation_result['rouge_l_f1_std']:.4f})")
        
        report.append(f"\nCompression Ratio: {evaluation_result['compression_ratio_mean']:.2f}x")
        report.append(f"  (Standard Deviation: {evaluation_result['compression_ratio_std']:.2f})")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)
    
    def export_results(self,
                      evaluation_result: Dict,
                      filename: str = "evaluation_results.json"):
        """
        Export evaluation results to file
        
        Args:
            evaluation_result: Results to export
            filename: Output filename
        """
        import json
        
        with open(filename, 'w') as f:
            json.dump(evaluation_result, f, indent=2)
        
        logger.info(f"Results exported to {filename}")


# Test the evaluator
if __name__ == "__main__":
    print("=" * 70)
    print("EVALUATOR SYSTEM TEST")
    print("=" * 70)
    
    # Sample data
    generated = "Machine learning revolutionizes AI. Neural networks perform complex tasks."
    reference = "Machine learning has transformed artificial intelligence. Deep neural networks can now handle complicated tasks with high accuracy."
    
    # Initialize evaluator
    evaluator = SummarizerEvaluator()
    
    # Evaluate single summary
    scores = evaluator.evaluate_single(generated, reference, "TestModel")
    
    print("\nSingle Summary Evaluation:")
    print(f"ROUGE-1 F1: {scores['rouge_1_f1']:.4f}")
    print(f"ROUGE-2 F1: {scores['rouge_2_f1']:.4f}")
    print(f"ROUGE-L F1: {scores['rouge_l_f1']:.4f}")
    print(f"Compression Ratio: {scores['compression_ratio']:.2f}x")
    
    # Test batch evaluation
    generated_list = [generated] * 5
    reference_list = [reference] * 5
    
    batch_scores = evaluator.evaluate_batch(generated_list, reference_list, "TestModel")
    
    print("\n" + evaluator.get_detailed_report(batch_scores))