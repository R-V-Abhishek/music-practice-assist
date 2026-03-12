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

from .raga_database import RagaInfo, get_raga_info
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
    """Finite State Machine for raga sequence validation"""
    
    def __init__(self, raga_info: RagaInfo):
        self.raga_info = raga_info
        self.reset()
    
    def reset(self):
        """Reset FSM to initial state"""
        self.current_direction = Direction.UNKNOWN
        self.arohana_position = 0
        self.avarohana_position = 0
        self.last_swara = None
    
    def validate_sequence(self, swara: str, direction: Direction) -> Optional[ErrorType]:
        """
        Validate if swara follows correct Arohana/Avarohana sequence.
        
        Args:
            swara: Current swara
            direction: Detected motion direction
            
        Returns:
            ErrorType if violation detected, None if valid
        """
        # Handle direction changes
        if direction != self.current_direction:
            self._handle_direction_change(direction)
        
        # Choose appropriate sequence based on direction
        if direction == Direction.ASCENDING:
            return self._validate_arohana_step(swara)
        elif direction == Direction.DESCENDING:
            return self._validate_avarohana_step(swara)
        else:
            # Neutral/unknown direction - allow any note in either sequence
            arohana_valid = self._is_swara_in_sequence(swara, self.raga_info.arohana)
            avarohana_valid = self._is_swara_in_sequence(swara, self.raga_info.avarohana)
            
            if not (arohana_valid or avarohana_valid):
                return ErrorType.SEQUENCE_VIOLATION
        
        self.last_swara = swara
        return None
    
    def _handle_direction_change(self, new_direction: Direction):
        """Handle direction change - reset appropriate FSM state"""
        self.current_direction = new_direction
        
        if new_direction == Direction.ASCENDING:
            # Reset to beginning of arohana, or find current position
            self._find_arohana_position(self.last_swara)
        elif new_direction == Direction.DESCENDING:
            # Reset to beginning of avarohana, or find current position
            self._find_avarohana_position(self.last_swara)
    
    def _validate_arohana_step(self, swara: str) -> Optional[ErrorType]:
        """Validate step in ascending direction"""
        arohana = self.raga_info.arohana
        
        # Find where we should be in the sequence
        expected_positions = []
        for i in range(self.arohana_position, len(arohana)):
            if arohana[i] == swara:
                expected_positions.append(i)
        
        if not expected_positions:
            # Swara not found in remaining arohana
            return ErrorType.SEQUENCE_VIOLATION
        
        # Allow forward motion (elision is OK in Carnatic music)
        new_position = min(expected_positions)
        if new_position >= self.arohana_position:
            self.arohana_position = new_position + 1
            return None
        else:
            # Backward motion in ascending direction
            return ErrorType.WRONG_DIRECTION
    
    def _validate_avarohana_step(self, swara: str) -> Optional[ErrorType]:
        """Validate step in descending direction"""
        avarohana = self.raga_info.avarohana
        
        # Find where we should be in the sequence
        expected_positions = []
        for i in range(self.avarohana_position, len(avarohana)):
            if avarohana[i] == swara:
                expected_positions.append(i)
        
        if not expected_positions:
            return ErrorType.SEQUENCE_VIOLATION
        
        # Allow forward motion in avarohana sequence
        new_position = min(expected_positions)
        if new_position >= self.avarohana_position:
            self.avarohana_position = new_position + 1
            return None
        else:
            return ErrorType.WRONG_DIRECTION
    
    def _find_arohana_position(self, swara: Optional[str]):
        """Find current position in arohana sequence"""
        if swara is None:
            self.arohana_position = 0
            return
        
        # Find the earliest occurrence of this swara
        for i, seq_swara in enumerate(self.raga_info.arohana):
            if seq_swara == swara:
                self.arohana_position = i + 1
                return
        
        self.arohana_position = 0
    
    def _find_avarohana_position(self, swara: Optional[str]):
        """Find current position in avarohana sequence"""
        if swara is None:
            self.avarohana_position = 0
            return
        
        for i, seq_swara in enumerate(self.raga_info.avarohana):
            if seq_swara == swara:
                self.avarohana_position = i + 1
                return
        
        self.avarohana_position = 0
    
    def _is_swara_in_sequence(self, swara: str, sequence: List[str]) -> bool:
        """Check if swara appears anywhere in the sequence"""
        return swara in sequence
    
    def get_expected_swaras(self, direction: Direction) -> List[str]:
        """Get list of expected next swaras for current FSM state"""
        if direction == Direction.ASCENDING:
            sequence = self.raga_info.arohana
            position = self.arohana_position
        elif direction == Direction.DESCENDING:
            sequence = self.raga_info.avarohana
            position = self.avarohana_position
        else:
            # Return all allowed swaras for neutral direction
            return list(set(self.raga_info.arohana + self.raga_info.avarohana))
        
        if position < len(sequence):
            return [sequence[position]]
        else:
            return []  # End of sequence reached

class RagaGrammarValidator:
    """
    Main raga grammar validation engine.
    
    Implements layered validation:
    1. Forbidden swara detection (immediate) 
    2. Sequence order validation (FSM)
    3. Direction-sensitive rule checking
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
        self.direction_detector = DirectionDetector()
        self.fsm = RagaGrammarFSM(self.raga_info)
        
        # Validation history
        self.events: List[ValidationEvent] = []
        self.start_time = time.time()
    
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
        direction = self.direction_detector.add_swara(swara)
        
        # Layer 1: Immediate forbidden swara check
        forbidden_error = self._check_forbidden_swara(swara, direction)
        if forbidden_error:
            event = ValidationEvent(
                timestamp_ms=timestamp_ms,
                swara=swara,
                octave=swara_result.octave,
                frequency_hz=0,  # Will be filled by caller
                cents_deviation=swara_result.cents_deviation,
                error_type=forbidden_error,
                direction=direction,
                description=f"Forbidden note {swara} in {direction.value} for {self.raga_name}",
                confidence=swara_result.confidence
            )
            self.events.append(event)
            return event
        
        # Layer 2: Sequence validation via FSM
        sequence_error = self.fsm.validate_sequence(swara, direction)
        expected_swaras = self.fsm.get_expected_swaras(direction)
        
        event = ValidationEvent(
            timestamp_ms=timestamp_ms,
            swara=swara,
            octave=swara_result.octave,
            frequency_hz=0,
            cents_deviation=swara_result.cents_deviation,
            error_type=sequence_error,
            direction=direction,
            expected_swara=expected_swaras[0] if expected_swaras else None,
            description="Valid swara" if not sequence_error else f"Sequence violation: {sequence_error.value}",
            confidence=swara_result.confidence
        )
        
        self.events.append(event)
        return event
    
    def _check_forbidden_swara(self, swara: str, direction: Direction) -> Optional[ErrorType]:
        """Check if swara is forbidden in current direction"""
        if direction == Direction.ASCENDING and swara in self.raga_info.varja_arohana:
            return ErrorType.FORBIDDEN_NOTE
        elif direction == Direction.DESCENDING and swara in self.raga_info.varja_avarohana:
            return ErrorType.FORBIDDEN_NOTE
        
        return None
    
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
        self.events.clear()
        self.start_time = time.time()
    
    def get_errors_only(self) -> List[ValidationEvent]:
        """Get only validation events with errors"""
        return [e for e in self.events if e.error_type is not None]