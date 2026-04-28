"""
Raga Grammar Validator - FSM-based Rule Engine

Core rule engine implementing:
1. Layer 1: Immediate forbidden swara detection (O(1) set lookup)  
2. Layer 2: Sequential Arohana/Avarohana FSM with direction detection
3. Real-time streaming validation with directional state tracking

Handles vakra ragas, zigzag patterns, and direction-sensitive forbidden notes.
"""

import numpy as np
from collections import deque
from typing import List, Optional, Dict, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import time

from .raga_database import RagaInfo, get_raga_info, SWARA_CENTS
from .swara_quantizer import SwaraResult, get_swara_ordinal

ENHARMONIC_PAIRS = [
    ('Ri2', 'Ga1'),
    ('Ri3', 'Ga2'),
    ('Dha2', 'Ni1'),
    ('Ni2', 'Dha3'),
]


def _resolve_enharmonic(swara: str, raga_info: RagaInfo) -> str:
    """Return raga-preferred enharmonic alias when only one name is used."""
    all_raga_swaras = set(raga_info.arohana) | set(raga_info.avarohana)
    for a, b in ENHARMONIC_PAIRS:
        if swara == a and b in all_raga_swaras and a not in all_raga_swaras:
            return b
        if swara == b and a in all_raga_swaras and b not in all_raga_swaras:
            return a
    return swara

class ErrorType(Enum):
    """Types of raga grammar violations"""
    FORBIDDEN_NOTE = "forbidden_note"
    SEQUENCE_VIOLATION = "sequence_violation" 
    WRONG_DIRECTION = "wrong_direction"
    UNEXPECTED_JUMP = "unexpected_jump"
    FORBIDDEN_PHRASE = "forbidden_phrase"

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
    matched_phrase: Optional[List[str]] = None
    secondary_error: Optional[ErrorType] = None  # Second violation when both FORBIDDEN_NOTE + SEQUENCE_VIOLATION apply

def _derive_vakra_skip_phrases(sequence: List[str]) -> List[List[str]]:
    """Derive forbidden skip phrases from a vakra arohana/avarohana sequence.

    At each zigzag point *b* in the triplet [a, b, c], the pair [a, c] is a
    forbidden skip — the musician must not jump directly from *a* to *c*,
    skipping the required vakra detour through *b*.

    The trailing "Sa" (octave boundary marker) is stripped before analysis so
    that the wrap-around Sa→Sa at the octave edge does not create a spurious
    zigzag.
    """
    if len(sequence) < 3:
        return []

    # Strip the trailing Sa (octave boundary marker)
    seq = sequence[:-1] if sequence[-1] == 'Sa' else list(sequence)

    if len(seq) < 3:
        return []

    # Convert to cent values
    cents = [SWARA_CENTS.get(s, -1) for s in seq]

    phrases: List[List[str]] = []
    for i in range(len(cents) - 2):
        c0, c1, c2 = cents[i], cents[i + 1], cents[i + 2]
        if c0 < 0 or c1 < 0 or c2 < 0:
            continue
        diff1 = c1 - c0
        diff2 = c2 - c1
        if diff1 * diff2 < 0:  # Direction reversal = zigzag
            a, c = seq[i], seq[i + 2]
            if a != c:  # Skip same-swara pairs (e.g. Ga1→Ri1→Ga1)
                phrases.append([a, c])
    return phrases


class PhraseBuffer:
    """Rolling buffer of recent swara names for phrase-level detection."""

    def __init__(self, maxlen: int = 10):
        self._buf: deque = deque(maxlen=maxlen)

    def push(self, swara: str) -> None:
        self._buf.append(swara)

    def reset(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)

    def __getitem__(self, idx: int) -> str:
        return self._buf[idx]


class ForbiddenPhraseDetector:
    """Detects forbidden skip phrases in a swara buffer."""

    def __init__(self, forbidden_phrases: List[List[str]]):
        # Store as set of tuples for O(1) lookup
        self._forbidden = {tuple(p) for p in forbidden_phrases}

    def check(self, buf: PhraseBuffer) -> Optional[List[str]]:
        """Return the forbidden phrase if the last two swaras match, else None."""
        if len(buf) < 2:
            return None
        pair = (buf[-2], buf[-1])
        if pair in self._forbidden:
            return list(pair)
        return None


class CharacteristicPhraseDetector:
    """Detects characteristic (prayoga) phrases via suffix-match on the swara buffer.

    Longest match wins so that when a short phrase is a suffix of a longer one,
    the more complete prayoga is reported.
    """

    def __init__(self, phrases: List[List[str]], maxlen: int = 20):
        self._by_length: Dict[int, Set[Tuple[str, ...]]] = {}
        for p in phrases:
            if 2 <= len(p) <= maxlen:
                key = tuple(p)
                self._by_length.setdefault(len(p), set()).add(key)
        # Pre-sorted lengths in decreasing order for longest-match-first iteration
        self._sorted_lengths: List[int] = sorted(self._by_length.keys(), reverse=True)

    def check(self, buf: PhraseBuffer) -> Optional[List[str]]:
        """Return the longest matching characteristic phrase, or None."""
        buf_len = len(buf)
        for L in self._sorted_lengths:
            if buf_len < L:
                continue
            tail = tuple(buf[i] for i in range(buf_len - L, buf_len))
            if tail in self._by_length[L]:
                return list(tail)
        return None


class PairwiseDirectionTracker:
    """Detects ascending/descending motion immediately between two distinct notes."""
    
    def __init__(self, neutral_timeout_ms: float = 1750.0):
        self._dir = Direction.NEUTRAL
        self._last_note_time: Optional[float] = None
        self._last_confirmed_swara: Optional[str] = None
        self._last_octave: Optional[int] = None
        self.neutral_timeout_ms = neutral_timeout_ms

    def update(self, swara: str, octave: int, timestamp_ms: float) -> Direction:
        swara = swara.split(' (')[0]  # strip '(Shruti N)' tag
        if self._last_confirmed_swara != swara or self._last_octave != octave:
            if self._last_confirmed_swara is not None and self._last_octave is not None:
                prev_cents = SWARA_CENTS.get(self._last_confirmed_swara)
                curr_cents = SWARA_CENTS.get(swara)
                if prev_cents is not None and curr_cents is not None:
                    prev_pos = self._last_octave * 1200 + prev_cents
                    curr_pos = octave * 1200 + curr_cents
                    if curr_pos > prev_pos:
                        self._dir = Direction.ASCENDING
                    elif curr_pos < prev_pos:
                        self._dir = Direction.DESCENDING
            
            self._last_confirmed_swara = swara
            self._last_octave = octave
            self._last_note_time = timestamp_ms

        if self._last_note_time is not None and (timestamp_ms - self._last_note_time) >= self.neutral_timeout_ms:
            self._dir = Direction.NEUTRAL
        
        return self._dir

    def tick(self, timestamp_ms: float) -> bool:
        """Returns True if timeout just fired"""
        if self._dir != Direction.NEUTRAL and self._last_note_time is not None:
            if (timestamp_ms - self._last_note_time) >= self.neutral_timeout_ms:
                self._dir = Direction.NEUTRAL
                return True
        return False
        
    def reset(self):
        self._dir = Direction.NEUTRAL
        self._last_note_time = None
        self._last_confirmed_swara = None
        self._last_octave = None

class SubsequenceTracker:
    """Tracks strict sequential traversal of arohana/avarohana with special phrase bypass."""
    def __init__(self, raga_info: RagaInfo, lookahead: int = 3):
        self.arohana = raga_info.arohana
        self.avarohana = raga_info.avarohana
        self.special_phrases = raga_info.special_phrases
        self.lookahead = lookahead
        self._aro_ptr = 0
        self._ava_ptr = 0
        
    def reset(self):
        self._aro_ptr = 0
        self._ava_ptr = 0
        
    def step(self, swara: str, prev_swara: Optional[str], direction: Direction, phrase_buf: PhraseBuffer) -> bool:
        if swara == prev_swara:
            return True
            
        # Check special phrase bypass first
        buf_list = [phrase_buf[i] for i in range(len(phrase_buf))] + [swara]
        for sp in self.special_phrases:
            if len(buf_list) >= len(sp):
                if buf_list[-len(sp):] == sp:
                    self._sync_pointers_to_swara(swara)
                    return True
                    
        is_valid = False
        
        # Arohana traversal
        if direction != Direction.DESCENDING:
            new_ptr = self._find_next_ptr(swara, self.arohana, self._aro_ptr)
            if new_ptr is not None:
                self._aro_ptr = new_ptr
                is_valid = True
                
        # Avarohana traversal
        if direction != Direction.ASCENDING:
            new_ptr = self._find_next_ptr(swara, self.avarohana, self._ava_ptr)
            if new_ptr is not None:
                self._ava_ptr = new_ptr
                is_valid = True

        if not is_valid:
            if prev_swara is None:
                self._sync_pointers_to_swara(swara)
                return True
            return False
            
        return True
        
    def _find_next_ptr(self, swara: str, sequence: List[str], current_ptr: int) -> Optional[int]:
        if not sequence:
            return None
        for i in range(1, self.lookahead + 1):
            next_idx = (current_ptr + i) % len(sequence)
            if sequence[next_idx] == swara:
                return next_idx
        return None
        
    def _sync_pointers_to_swara(self, swara: str):
        for seq, ptr_attr in [(self.arohana, '_aro_ptr'), (self.avarohana, '_ava_ptr')]:
            current_ptr = getattr(self, ptr_attr)
            best_idx = current_ptr
            min_dist = float('inf')
            found = False
            for i, s in enumerate(seq):
                if s == swara:
                    found = True
                    dist = min(abs(i - current_ptr), len(seq) - abs(i - current_ptr))
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = i
            if found:
                setattr(self, ptr_attr, best_idx)

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
        
        # Still apply directional varja: a note varja in BOTH directions is always forbidden.
        # A note varja only in one direction is allowed in NEUTRAL (ambiguous motion).
        if swara in self.raga_info.varja_arohana and swara in self.raga_info.varja_avarohana:
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
        self.direction_tracker = PairwiseDirectionTracker(neutral_timeout_ms=1750.0)
        
        # Conditional grammar mode
        self._has_special_phrases = len(self.raga_info.special_phrases) > 0
        self._mode = "full" if self._has_special_phrases else "simple"
        
        self.fsm = RagaGrammarFSM(self.raga_info)
        self.subsequence_tracker = SubsequenceTracker(self.raga_info)
        
        # Octave-aware pairwise direction tracking for validation
        self._prev_swara_result: Optional[SwaraResult] = None

        # Phase 3: Forbidden phrase detection
        _forbidden_phrases = (
            _derive_vakra_skip_phrases(self.raga_info.arohana)
            + _derive_vakra_skip_phrases(self.raga_info.avarohana)
        )
        self.phrase_buffer = PhraseBuffer(maxlen=20)
        self.phrase_detector = ForbiddenPhraseDetector(_forbidden_phrases)

        # Phase 4: Characteristic phrase recognition
        _char_phrases = self.raga_info.special_phrases + self.raga_info.characteristic_phrases
        self.characteristic_detector = CharacteristicPhraseDetector(_char_phrases, maxlen=20)

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
        swara = swara.split(' (')[0]  # strip '(Shruti N)' tag if present
        swara = _resolve_enharmonic(swara, self.raga_info)
        swara_result = SwaraResult(
            swara=swara,
            octave=swara_result.octave,
            cents_deviation=swara_result.cents_deviation,
            confidence=swara_result.confidence,
        )
        
        # UI display direction
        ui_direction = self.direction_tracker.update(swara, swara_result.octave, timestamp_ms)
        
        # Octave-aware pairwise direction for validation decisions
        validation_direction = self._get_step_direction(self._prev_swara_result, swara_result)
        
        error = None
        expected_swaras = []
        matched_phrase = None
        description = "Valid swara"
        
        if self._mode == "simple":
            # Mode 1: Absolute forbidden note only
            secondary_error = None
            if swara not in self.fsm.arohana_set and swara not in self.fsm.avarohana_set:
                error = ErrorType.FORBIDDEN_NOTE
                description = f"Forbidden note {swara} in {self.raga_name}"
            expected_swaras = list(self.fsm.arohana_set | self.fsm.avarohana_set)
        else:
            # Mode 2: Full grammar
            secondary_error = None

            # 1. Push phrase_buffer FIRST so special-phrase bypass has full context
            self.phrase_buffer.push(swara)

            # 2. Special phrase bypass — if current swara completes a special phrase,
            #    skip all varja and sequence checks for this step
            buf_list = [self.phrase_buffer[i] for i in range(len(self.phrase_buffer))]
            in_special_phrase = False
            for sp in self.raga_info.special_phrases:
                if len(buf_list) >= len(sp) and buf_list[-len(sp):] == sp:
                    in_special_phrase = True
                    if self._prev_swara_result is not None:
                        self.subsequence_tracker._sync_pointers_to_swara(swara)
                    break

            if not in_special_phrase:
                # 3. Direction-based varja check
                error = self.fsm.validate_sequence(swara, validation_direction)
                expected_swaras = self.fsm.get_expected_swaras(validation_direction)
                if error:
                    description = f"Forbidden note {swara} in {validation_direction.value} for {self.raga_name}"

                # 4. Sequential subsequence check — always run to track pointer
                prev_s = self._prev_swara_result.swara if self._prev_swara_result else None
                is_valid_seq = self.subsequence_tracker.step(swara, prev_s, validation_direction, self.phrase_buffer)
                if not is_valid_seq:
                    seq_err = ErrorType.SEQUENCE_VIOLATION
                    seq_desc = f"Sequence violation at {swara} in {validation_direction.value}"
                    if error is None:
                        error = seq_err
                        description = seq_desc
                    else:
                        # Both apply — stash sequence violation as secondary
                        secondary_error = seq_err
                        description = f"{description}; {seq_desc}"

                # 5. Forbidden skip phrase check (vakra skips)
                if not error:
                    matched = self.phrase_detector.check(self.phrase_buffer)
                    if matched:
                        error = ErrorType.FORBIDDEN_PHRASE
                        description = f"Forbidden skip {matched[0]}\u2192{matched[1]} in {self.raga_name}"
                    else:
                        matched_char = self.characteristic_detector.check(self.phrase_buffer)
                        if matched_char:
                            matched_phrase = matched_char
            else:
                # Inside special phrase — run subsequence sync, no errors
                expected_swaras = self.fsm.get_expected_swaras(validation_direction)
                secondary_error = None

        event = ValidationEvent(
            timestamp_ms=timestamp_ms,
            swara=swara,
            octave=swara_result.octave,
            frequency_hz=0,  # Will be filled by caller
            cents_deviation=swara_result.cents_deviation,
            error_type=error,
            direction=ui_direction,
            expected_swara=expected_swaras[0] if expected_swaras else None,
            description=description,
            confidence=swara_result.confidence,
            matched_phrase=matched_phrase,
            secondary_error=secondary_error,
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
    
    def reset_phrase_buffer(self):
        """Reset phrase buffer and pairwise direction state between melodic phrases."""
        self.phrase_buffer.reset()
        self._prev_swara_result = None
        self.subsequence_tracker.reset()
        
    def tick_direction(self, timestamp_ms: float):
        """Called every voiced frame to update neutral timeout"""
        if self.direction_tracker.tick(timestamp_ms):
            self.reset_phrase_buffer()

    def reset(self):
        """Reset validator state for new analysis"""
        self.direction_tracker.reset()
        self.fsm.reset()
        self.subsequence_tracker.reset()
        self._prev_swara_result = None
        self.phrase_buffer.reset()
        self.events.clear()
        self.start_time = time.time()
    
    def get_errors_only(self) -> List[ValidationEvent]:
        """Get only validation events with errors"""
        return [e for e in self.events if e.error_type is not None]