"""
Raga Grammar Validation System

A comprehensive rule-based engine for validating Carnatic music performance
against raga-specific Arohana/Avarohana constraints and forbidden swara detection.

Components:
- raga_database: Knowledge base of 34 common ragas + 72 Melakarta ragas
- swara_quantizer: Hz → swara name conversion with Sa-relative cents
- grammar_validator: FSM-based sequence validation and forbidden note detection
- pitch_pipeline: Real-time pYIN integration with streaming validation
- feedback_generator: Human-readable error messages
"""

from .raga_database import RAGA_DB, get_raga_info, get_melakarta_raga
from .swara_quantizer import SwaraQuantizer
from .grammar_validator import RagaGrammarValidator, ValidationEvent
from .pitch_pipeline import RealTimeGrammarPipeline
from .feedback_generator import FeedbackGenerator

__version__ = "1.0.0"
__all__ = [
    "RAGA_DB",
    "get_raga_info", 
    "get_melakarta_raga",
    "SwaraQuantizer",
    "RagaGrammarValidator", 
    "ValidationEvent",
    "RealTimeGrammarPipeline",
    "FeedbackGenerator"
]