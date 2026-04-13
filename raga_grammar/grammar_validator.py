"""
Raga Grammar Validator - FSM-based Rule Engine

Core rule engine implementing:
1. Layer 1: Immediate forbidden swara detection (O(1) set lookup)  
2. Layer 2: Sequential Arohana/Avarohana FSM with direction detection
3. Real-time streaming validation with directional state tracking

Handles vakra ragas, zigzag patterns, and direction-sensitive forbidden notes.
"""

import numpy as np
from typing import List, Optional, Dict, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import time

from .raga_database import RagaInfo, get_raga_info, SWARA_CENTS
from .swara_quantizer import SwaraResult, get_swara_ordinal

class ErrorType(Enum):
    """Types of raga grammar violations"""
    FORBIDDEN_NOTE = "forbidden_note"
    SEQUENCE_VIOLATION = "sequence_violation" 
    WRONG_DIRECTION = "wrong_direction"
    UNEXPECTED_JUMP = "unexpected_jump"

class Direction(Enum):
    """Musical direction states"""
    ASCENDING = "ascending"
    DESCENDING = "descending" 
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"

@dataclass
class ValidationEvent:
    """Single validation event with error details"""
    timestamp_ms: float
    swara: str
    octave: int
    frequency_hz: float
    cents_deviation: float
    error_type: Optional[ErrorType]
    direction: Direction
    expected_swara: Optional[str] = None
    description: str = ""
    confidence: float = 1.0

class DirectionDetector:
    """Detects ascending/descending motion from swara sequence"""
    
    def __init__(self, history_length: int = 3):
        self.history: List[str] = []
        self.history_length = history_length
    
    def add_swara(self, swara: str) -> Direction:
        """Add swara and detect current direction"""
        self.history.append(swara)
        if len(self.history) > self.history_length:
            self.history.pop(0)
        
        if len(self.history) < 2:
            return Direction.UNKNOWN
        
        # Get ordinal positions for direction calculation
        ordinals = [get_swara_ordinal(s) for s in self.history if get_swara_ordinal(s) >= 0]
        if len(ordinals) < 2:
            return Direction.UNKNOWN
        
        # Analyze recent motion (weighted toward recent notes)
        direction_score = 0
        for i in range(1, len(ordinals)):
            diff = ordinals[i] - ordinals[i-1]
            weight = i / len(ordinals)  # Recent notes have higher weight
            if diff > 0:
                direction_score += weight
            elif diff < 0:
                direction_score -= weight
        
        # Classify direction
        if direction_score > 0.3:
            return Direction.ASCENDING
        elif direction_score < -0.3:
            return Direction.DESCENDING
        else:
            return Direction.NEUTRAL
    
    def reset(self):
        """Reset direction history"""
        self.history.clear()

class RagaGrammarFSM:
    """Stateless per-step membership validator for raga swaras.
    
    Pure function of (swara, direction) → valid?  No position pointers.
    Handles vakra ragas naturally via set membership; phrase-level vakra
    constraints are deferred to Phase 3 forbidden-phrase detection.
    """
    
    def __init__(self, raga_info: RagaInfo):
        self.raga_info = raga_info
        # Pre-compute membership sets for O(1) lookup
        self.arohana_set: Set[str] = set(raga_info.arohana)
        self.avarohana_set: Set[str] = set(raga_info.avarohana)
    
    def reset(self):
        """No-op — FSM is stateless. Kept for API compatibility."""
        pass
    
    def validate_sequence(self, swara: str, direction: Direction) -> Optional[ErrorType]:
        """Validate swara membership for the given step direction.
        
        ASCENDING  → forbidden if swara in varja_arohana
        DESCENDING → forbidden if swara in varja_avarohana
        NEUTRAL    → must appear in arohana OR avarohana set
        UNKNOWN    → always pass (very first note)
        
        Args:
            swara: Current swara name
            direction: Octave-aware pairwise direction for this step
            
        Returns:
            ErrorType.FORBIDDEN_NOTE on violation, None if valid
        """
        if direction == Direction.UNKNOWN:
            return None
        
        if direction == Direction.ASCENDING:
            if swara in self.raga_info.varja_arohana:
                return ErrorType.FORBIDDEN_NOTE
            return None
        
        if direction == Direction.DESCENDING:
            if swara in self.raga_info.varja_avarohana:
                return ErrorType.FORBIDDEN_NOTE
            return None
        
        # NEUTRAL — swara must belong to at least one direction's set
        if swara not in self.arohana_set and swara not in self.avarohana_set:
            return ErrorType.FORBIDDEN_NOTE
        
        return None
    
    def get_expected_swaras(self, direction: Direction) -> List[str]:
        """Get list of allowed swaras for the given direction."""
        if direction == Direction.ASCENDING:
            return list(self.arohana_set)
        elif direction == Direction.DESCENDING:
            return list(self.avarohana_set)
        else:
            return list(self.arohana_set | self.avarohana_set)

class RagaGrammarValidator:
    """
    Main raga grammar validation engine.
    
    Single-layer validation via stateless per-step membership FSM.
    Direction is computed octave-aware from consecutive SwaraResults.
    DirectionDetector is retained for UI display direction only.
    """
    
    def __init__(self, raga_name: str):
        """
        Initialize validator for a specific raga.
        
        Args:
            raga_name: Name of raga to validate against
            
        Raises:
            ValueError: If raga not found in database
        """
        self.raga_info = get_raga_info(raga_name)
        if not self.raga_info:
            raise ValueError(f"Raga '{raga_name}' not found in database")
        
        self.raga_name = raga_name
        self.direction_detector = DirectionDetector()  # UI display direction only
        self.fsm = RagaGrammarFSM(self.raga_info)
        
        # Octave-aware pairwise direction tracking for validation
        self._prev_swara_result: Optional[SwaraResult] = None
        
        # Validation history
        self.events: List[ValidationEvent] = []
        self.start_time = time.time()
    
    def _get_step_direction(self, prev: Optional[SwaraResult], curr: SwaraResult) -> Direction:
        """
        Octave-aware pairwise direction between two consecutive SwaraResults.
        
        Converts each swara to an absolute pitch position using:
            pitch_pos = octave * 1200 + SWARA_CENTS[swara]
        and compares to determine melodic direction.
        
        Args:
            prev: Previous SwaraResult (None at start of session)
            curr: Current SwaraResult
            
        Returns:
            Direction.ASCENDING / DESCENDING / NEUTRAL / UNKNOWN
        """
        if prev is None:
            return Direction.UNKNOWN
        
        prev_cents = SWARA_CENTS.get(prev.swara)
        curr_cents = SWARA_CENTS.get(curr.swara)
        
        if prev_cents is None or curr_cents is None:
            return Direction.UNKNOWN
        
        prev_pos = prev.octave * 1200 + prev_cents
        curr_pos = curr.octave * 1200 + curr_cents
        
        if curr_pos > prev_pos:
            return Direction.ASCENDING
        elif curr_pos < prev_pos:
            return Direction.DESCENDING
        else:
            return Direction.NEUTRAL
    
    def validate_swara(self, swara_result: SwaraResult, 
                      timestamp_ms: Optional[float] = None) -> ValidationEvent:
        """
        Validate a single swara result against raga rules.
        
        Args:
            swara_result: Quantized swara with deviation info
            timestamp_ms: Optional timestamp, auto-generated if None
            
        Returns:
            ValidationEvent with error details or success confirmation
        """
        if timestamp_ms is None:
            timestamp_ms = (time.time() - self.start_time) * 1000
        
        swara = swara_result.swara
        
        # UI display direction (phrase-level, from DirectionDetector)
        ui_direction = self.direction_detector.add_swara(swara)
        
        # Octave-aware pairwise direction for validation decisions
        validation_direction = self._get_step_direction(self._prev_swara_result, swara_result)
        
        # Unified membership validation (forbidden-note + direction check in one step)
        error = self.fsm.validate_sequence(swara, validation_direction)
        expected_swaras = self.fsm.get_expected_swaras(validation_direction)
        
        event = ValidationEvent(
            timestamp_ms=timestamp_ms,
            swara=swara,
            octave=swara_result.octave,
            frequency_hz=0,  # Will be filled by caller
            cents_deviation=swara_result.cents_deviation,
            error_type=error,
            direction=ui_direction,
            expected_swara=expected_swaras[0] if expected_swaras else None,
            description=(
                "Valid swara" if not error
                else f"Forbidden note {swara} in {validation_direction.value} for {self.raga_name}"
            ),
            confidence=swara_result.confidence
        )
        
        self.events.append(event)
        self._prev_swara_result = swara_result
        return event
    
    def get_validation_summary(self) -> Dict:
        """Get summary of all validation events"""
        total_events = len(self.events)
        errors = [e for e in self.events if e.error_type is not None]
        error_types = {}
        
        for error in errors:
            error_type = error.error_type.value
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'raga': self.raga_name,
            'total_swaras': total_events,
            'total_errors': len(errors),
            'error_rate': len(errors) / max(total_events, 1),
            'error_breakdown': error_types,
            'duration_ms': self.events[-1].timestamp_ms if self.events else 0
        }
    
    def reset(self):
        """Reset validator state for new analysis"""
        self.direction_detector.reset()
        self.fsm.reset()
        self._prev_swara_result = None
        self.events.clear()
        self.start_time = time.time()
    
    def get_errors_only(self) -> List[ValidationEvent]:
        """Get only validation events with errors"""
        return [e for e in self.events if e.error_type is not None]