"""
Swara Quantizer - Convert Hz to Carnatic Swara Names

Maps continuously pitched audio to discrete 12-swara labels using Sa-relative
cents with tolerance windows. Handles octave folding and provides
deviation measurements for apashruthi detection.

Integration: Uses CARNATIC_RATIOS from tonic_sa_detection.py for consistent
frequency-to-swara mapping across the entire system.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class SwaraResult:
    """Result of Hz → Swara quantization"""
    swara: str              # Quantized swara name (e.g., "Ri2", "Ga2")
    octave: int             # Octave number (0 = Sa octave, 1 = upper Sa, -1 = lower)
    cents_deviation: float  # Deviation from perfect swara in cents
    confidence: float       # Confidence score (0-1) based on proximity

class SwaraQuantizer:
    """
    Converts Hz frequencies to Carnatic swara names using Sa-relative ratios.
    
    Uses the same CARNATIC_RATIOS mapping as tonic_sa_detection.py for consistency.
    Provides ±25 cent tolerance windows around each swara position.
    """
    
    # Carnatic note ratios from Sa (imported from tonic_sa_detection.py logic)
    CARNATIC_RATIOS = {
        'Sa': 1.0, 'Ri1': 256/243, 'Ri2': 9/8, 'Ga2': 32/27,
        'Ga3': 5/4, 'Ma1': 4/3, 'Ma2': 45/32, 'Pa': 3/2,
        'Dha1': 128/81, 'Dha2': 5/3, 'Ni2': 16/9, 'Ni3': 15/8,
    }
    
    # Convert ratios to cents for easier comparison
    SWARA_CENTS = {}
    for swara, ratio in CARNATIC_RATIOS.items():
        SWARA_CENTS[swara] = 1200 * np.log2(ratio)
    
    # Tolerance window for live singing (gamakas / microphone noise)
    TOLERANCE_CENTS = 50.0

    # If a low-confidence Ni3 sits very close to an octave Sa anchor,
    # prefer Sa to suppress harmonic-doubling confusion.
    SA_ANCHOR_SNAP_CENTS = 75.0
    NI3_SNAP_CONFIDENCE_MAX = 0.65

    # Supported live range relative to tonic:
    # ati anumaandra Sa (Sa/4) to ati taara Sa (4*Sa)
    MIN_OCTAVE_OFFSET = -2
    MAX_OCTAVE_OFFSET = 2
    
    def __init__(self, sa_frequency: float):
        """
        Initialize quantizer with detected Sa frequency.
        
        Args:
            sa_frequency: Tonic Sa frequency in Hz (from TonicSaDetector)
        """
        self.sa_frequency = sa_frequency
        self.min_supported_frequency = self.sa_frequency * (2 ** self.MIN_OCTAVE_OFFSET)
        self.max_supported_frequency = self.sa_frequency * (2 ** self.MAX_OCTAVE_OFFSET)
        
        # Pre-compute swara frequency boundaries for fast lookup
        self._swara_boundaries = {}
        for swara, cents in self.SWARA_CENTS.items():
            freq = sa_frequency * (2 ** (cents / 1200))
            self._swara_boundaries[swara] = {
                'freq': freq,
                'cents': cents,
                'min_cents': cents - self.TOLERANCE_CENTS,
                'max_cents': cents + self.TOLERANCE_CENTS
            }

    def get_supported_frequency_range(self) -> Tuple[float, float]:
        """Return supported quantization range in Hz."""
        return self.min_supported_frequency, self.max_supported_frequency

    def is_frequency_in_range(self, frequency: float) -> bool:
        """Check whether an input frequency is inside supported live range."""
        if frequency <= 0 or not np.isfinite(frequency):
            return False
        return self.min_supported_frequency <= frequency <= self.max_supported_frequency

    def _normalize_to_tonic_band(self, frequency: float) -> Tuple[Optional[float], int]:
        """
        Normalize frequency into [Sa, upper Sa] tonic band via octave folding.
        
        All input frequencies are transposed into the fundamental octave [Sa, 2×Sa)
        for direct ratio matching. Note: upper Sa boundary (2×Sa) is folded to Sa 
        of the next octave.
        
        For example, with Sa=260 Hz:
        - 260 Hz → (260 Hz, octave 0) = Sa
        - 390 Hz → (390 Hz, octave 0) = Pa  
        - 520 Hz → (260 Hz, octave 1) = Upper Sa, represented as Sa+1
        - 130 Hz → (260 Hz, octave -1) = Lower Sa, represented as Sa-1

        Args:
            frequency: Input frequency in Hz
            
        Returns:
            (normalized_frequency, octave_shift)
            - normalized_frequency: frequency folded into [Sa, 2×Sa) band (Hz)
            - octave_shift: number of octaves traversed
        """
        if frequency <= 0 or not np.isfinite(frequency):
            return None, 0

        f = float(frequency)
        octave = 0
        
        # Safety limit: prevent infinite loops
        max_iterations = 10
        iteration = 0
        
        upper_boundary = 2.0 * self.sa_frequency

        # Fold DOWN into [Sa, 2*Sa) if at or above upper boundary
        while f >= upper_boundary and iteration < max_iterations:
            f /= 2.0
            octave += 1
            iteration += 1

        # Fold UP into [Sa, 2*Sa) if below Sa
        iteration = 0
        while f < self.sa_frequency and iteration < max_iterations:
            f *= 2.0
            octave -= 1
            iteration += 1

        # Final safety check: ensure within expected range [Sa, 2*Sa)
        if not (self.sa_frequency * 0.99 <= f < 2.01 * self.sa_frequency):
            return None, 0

        return f, octave

    def to_swara_tonic_band(self, frequency: float, tolerance_cents: Optional[float] = None) -> Optional[SwaraResult]:
        """
        Live-mode swara mapping: Direct tonic-ratio matching in [Sa, 2×Sa] band.
        
        **ALGORITHM:**
        1. Normalize input frequency to [Sa, 2×Sa] via octave folding
        2. For each of the 12 Carnatic swaras, calculate its absolute frequency
           using: swara_freq = Sa × ratio
        3. Compute deviation from each swara in CENTS (1200 × log2(f_input / f_swara))
        4. Find the closest match (minimum absolute deviation)
        5. Accept only if deviation < TOLERANCE_CENTS (35 cents = ~0.5 semitones)
        6. Return SwaraResult with confidence = 1 - (deviation / tolerance)
        
        This approach is **bulletproof** because:
        - Tonic-ratio basis means no frequency-generic algorithms (pure math)
        - Strict tolerance rejects out-of-tune notes
        - Confidence score reflects how close to theoretical swara
        - Normalized band prevents octave confusion
        
        Args:
            frequency: Input frequency in Hz from pitch detector
            
        Returns:
            SwaraResult with swara name, octave, deviation, confidence
            None if no match found within tolerance
        """
        if not self.is_frequency_in_range(frequency):
            return None

        # Step 1: Normalize to [Sa, 2×Sa] band
        norm_freq, octave = self._normalize_to_tonic_band(frequency)
        if norm_freq is None:
            return None

        tol = float(self.TOLERANCE_CENTS if tolerance_cents is None else tolerance_cents)

        # Double-check bounds (redundant safety)
        if norm_freq < (0.99 * self.sa_frequency) or norm_freq > (2.01 * self.sa_frequency):
            return None

        # Step 2 & 3: Calculate all 12 swara frequencies and deviations
        best_swara = None
        best_dev_cents = float("inf")
        best_raw_deviation = 0.0

        for swara, ratio in self.CARNATIC_RATIOS.items():
            # Expected frequency for this swara
            swara_freq_hz = self.sa_frequency * ratio
            
            # Deviation in cents: negative = flat, positive = sharp
            dev_cents = 1200.0 * np.log2(norm_freq / swara_freq_hz)
            # Wrap to nearest equivalent within one octave so frequencies near
            # upper Sa (2*Sa - eps) can still map to Sa, not Ni3.
            while dev_cents > 600.0:
                dev_cents -= 1200.0
            while dev_cents < -600.0:
                dev_cents += 1200.0
            abs_dev_cents = abs(dev_cents)
            
            # Step 4: Track best match
            if abs_dev_cents < best_dev_cents:
                best_dev_cents = abs_dev_cents
                best_swara = swara
                best_raw_deviation = dev_cents

        # Step 5: Tolerance check
        if best_swara is None or best_dev_cents > tol:
            return None

        # Step 6: Confidence score
        confidence = max(0.0, 1.0 - (best_dev_cents / tol))

        # Correct common low-register confusion: base Sa harmonics can be
        # detected slightly below upper Sa and appear as low-confidence Ni3.
        if best_swara == 'Ni3' and confidence < self.NI3_SNAP_CONFIDENCE_MAX:
            sa_octave = int(np.round(np.log2(float(frequency) / self.sa_frequency)))
            sa_anchor_hz = self.sa_frequency * (2 ** sa_octave)
            sa_anchor_dev = 1200.0 * np.log2(float(frequency) / sa_anchor_hz)

            if abs(sa_anchor_dev) <= self.SA_ANCHOR_SNAP_CENTS:
                sa_conf = max(0.0, 1.0 - (abs(sa_anchor_dev) / self.SA_ANCHOR_SNAP_CENTS))
                return SwaraResult(
                    swara='Sa',
                    octave=sa_octave,
                    cents_deviation=float(sa_anchor_dev),
                    confidence=float(sa_conf),
                )
        
        return SwaraResult(
            swara=best_swara,
            octave=octave,
            cents_deviation=float(best_raw_deviation),
            confidence=float(confidence),
        )
    
    def to_swara(self, frequency: float) -> Optional[SwaraResult]:
        """
        Convert a frequency (Hz) to the nearest Carnatic swara.
        
        Args:
            frequency: Input frequency in Hz
            
        Returns:
            SwaraResult or None if frequency is too far from any swara
        """
        if frequency <= 0:
            return None
        
        # Convert to Sa-relative cents, handling octaves
        sa_relative_cents = 1200 * np.log2(frequency / self.sa_frequency)
        
        # Fold into one octave (0-1200 cents) to find base swara
        octave = int(sa_relative_cents // 1200)
        folded_cents = sa_relative_cents % 1200
        
        # Handle negative frequencies (below Sa)
        if sa_relative_cents < 0:
            octave = int(sa_relative_cents // 1200) - 1
            folded_cents = sa_relative_cents - (octave * 1200)
        
        # Find closest swara within tolerance
        best_swara = None
        min_deviation = float('inf')
        final_deviation = 0
        
        for swara, boundaries in self._swara_boundaries.items():
            deviation = abs(folded_cents - boundaries['cents'])
            
            # Handle wraparound at octave boundary (e.g., Ni2 ≈ upper Sa)
            deviation_wrapped = abs(folded_cents - (boundaries['cents'] + 1200))
            
            # Choose the smaller deviation
            if deviation_wrapped < deviation:
                actual_deviation = deviation_wrapped
                raw_deviation = folded_cents - (boundaries['cents'] + 1200)
            else:
                actual_deviation = deviation
                raw_deviation = folded_cents - boundaries['cents']
            
            if actual_deviation <= self.TOLERANCE_CENTS and actual_deviation < min_deviation:
                min_deviation = actual_deviation
                best_swara = swara
                final_deviation = raw_deviation
        
        if best_swara is None:
            return None  # No swara within tolerance
        
        # Compute confidence based on proximity (closer = higher confidence)
        confidence = max(0, 1 - (min_deviation / self.TOLERANCE_CENTS))
        
        return SwaraResult(
            swara=best_swara,
            octave=octave,
            cents_deviation=final_deviation,
            confidence=confidence
        )
    
    def to_swara_sequence(self, frequencies: np.ndarray, 
                         min_confidence: float = 0.5) -> list[SwaraResult]:
        """
        Convert a sequence of frequencies to swara sequence.
        
        Args:
            frequencies: Array of frequencies in Hz
            min_confidence: Minimum confidence to include a result
            
        Returns:
            List of SwaraResult objects that meet confidence threshold
        """
        results = []
        for freq in frequencies:
            if np.isnan(freq) or freq <= 0:
                continue
            
            result = self.to_swara(freq)
            if result and result.confidence >= min_confidence:
                results.append(result)
        
        return results
    
    def get_swara_frequency(self, swara: str, octave: int = 0) -> float:
        """
        Get the ideal frequency for a given swara in a specific octave.
        
        Args:
            swara: Swara name (e.g., "Ri2", "Ma1")
            octave: Octave number (0 = Sa octave)
            
        Returns:
            Frequency in Hz
        """
        if swara not in self.CARNATIC_RATIOS:
            raise ValueError(f"Unknown swara: {swara}")
        
        base_ratio = self.CARNATIC_RATIOS[swara]
        octave_multiplier = 2 ** octave
        return self.sa_frequency * base_ratio * octave_multiplier
    
    def get_apashruthi_error(self, frequency: float) -> Optional[float]:
        """
        Calculate apashruthi (pitch error) in cents from nearest swara.
        
        Args:
            frequency: Input frequency in Hz
            
        Returns:
            Deviation in cents (positive = sharp, negative = flat) or None
        """
        result = self.to_swara(frequency)
        return result.cents_deviation if result else None
    
    def is_pitch_accurate(self, frequency: float, tolerance_cents: float = 10) -> bool:
        """
        Check if a frequency is within acceptable pitch tolerance.
        
        Args:
            frequency: Input frequency in Hz
            tolerance_cents: Acceptable deviation in cents
            
        Returns:
            True if pitch is within tolerance
        """
        error = self.get_apashruthi_error(frequency)
        return error is not None and abs(error) <= tolerance_cents

def get_swara_ordinal(swara: str) -> int:
    """
    Get numeric position of swara for sequence validation.
    
    Args:
        swara: Swara name
        
    Returns:
        Ordinal position (0-11), used for detecting ascending/descending motion
    """
    swara_order = ['Sa', 'Ri1', 'Ri2', 'Ga1', 'Ga2', 'Ma1', 'Ma2', 
                   'Pa', 'Dha1', 'Dha2', 'Ni1', 'Ni2']
    try:
        return swara_order.index(swara)
    except ValueError:
        return -1  # Unknown swara