"""
Swara Quantizer - Convert Hz to Carnatic Swara Names

Maps continuously pitched audio to discrete 12-swara labels using Sa-relative
cents with ±25 cent tolerance windows. Handles octave folding and provides
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
        'Sa': 1.0, 'Ri1': 256/243, 'Ri2': 9/8, 'Ga1': 32/27,
        'Ga2': 5/4, 'Ma1': 4/3, 'Ma2': 45/32, 'Pa': 3/2,
        'Dha1': 128/81, 'Dha2': 5/3, 'Ni1': 16/9, 'Ni2': 15/8,
    }
    
    # Convert ratios to cents for easier comparison
    SWARA_CENTS = {}
    for swara, ratio in CARNATIC_RATIOS.items():
        SWARA_CENTS[swara] = 1200 * np.log2(ratio)
    
    # Tolerance window: ±25 cents (half of smallest interval ~50 cents)
    TOLERANCE_CENTS = 25.0
    
    def __init__(self, sa_frequency: float):
        """
        Initialize quantizer with detected Sa frequency.
        
        Args:
            sa_frequency: Tonic Sa frequency in Hz (from TonicSaDetector)
        """
        self.sa_frequency = sa_frequency
        
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