"""
Models package for text summarization
Contains implementations of various summarization algorithms
"""

# Optional imports - import only what you need to avoid loading heavy dependencies
__all__ = [
    'BaseSummarizer',
    'TextRankSummarizer',
    'BARTSummarizer',
    'PEGASUSSummarizer'
]

# Lazy imports - import classes when accessed via package
def __getattr__(name):
    if name == 'BaseSummarizer':
        from .base_summarizer import BaseSummarizer
        return BaseSummarizer
    elif name == 'TextRankSummarizer':
        from .textrank import TextRankSummarizer
        return TextRankSummarizer
    elif name == 'BARTSummarizer':
        from .bart import BARTSummarizer
        return BARTSummarizer
    elif name == 'PEGASUSSummarizer':
        from .pegasus import PEGASUSSummarizer
        return PEGASUSSummarizer
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

