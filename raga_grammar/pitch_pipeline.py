"""
Pitch Pipeline - Real-Time Grammar Validation Integration

Integrates with existing tonic_sa_detection.py to provide end-to-end pipeline:
Audio → Sa Detection → pYIN Pitch Tracking → Swara Quantization → Grammar Validation

Supports both real-time streaming (23ms per frame) and offline file analysis.
"""

import numpy as np
import librosa
import time
from typing import List, Dict, Optional, Generator, Tuple
from dataclasses import dataclass

# Import from existing tonic detection system
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tonic_sa_detection import TonicSaDetector

# Import grammar validation components
from .swara_quantizer import SwaraQuantizer, SwaraResult
from .grammar_validator import RagaGrammarValidator, ValidationEvent
from .raga_database import get_raga_info

@dataclass
class FrameResult:
    """Result for a single audio frame"""
    timestamp_ms: float
    frequency_hz: float
    voiced: bool
    voicing_prob: float
    swara_result: Optional[SwaraResult]
    validation_event: Optional[ValidationEvent]

class RealTimeGrammarPipeline:
    """
    Real-time audio analysis pipeline for raga grammar validation.
    
    Pipeline stages:
    1. Audio framing (2048 samples, hop=512, ~23ms at 22050 Hz)
    2. Tonic Sa detection (Carnatic HPS ensemble)
    3. pYIN pitch tracking per frame
    4. Swara quantization using Sa reference
    5. Grammar validation via FSM
    """
    
    def __init__(self, raga_name: str, sr: int = 22050):
        """
        Initialize pipeline for specific raga.
        
        Args:
            raga_name: Raga to validate against
            sr: Sample rate (Hz)
            
        Raises:
            ValueError: If raga not found
        """
        self.raga_name = raga_name
        self.sr = sr
        
        # Verify raga exists
        if not get_raga_info(raga_name):
            raise ValueError(f"Raga '{raga_name}' not found in database")
        
        # Audio processing parameters (optimized for <50ms latency)
        self.frame_length = 2048  # ~93ms at 22050 Hz
        self.hop_length = 512     # ~23ms hop for real-time streaming
        
        # Initialize components
        self.tonic_detector = TonicSaDetector(sr=sr)
        self.swara_quantizer: Optional[SwaraQuantizer] = None
        self.grammar_validator: Optional[RagaGrammarValidator] = None
        self.sa_frequency: Optional[float] = None
        
    def detect_tonic_from_file(self, audio_path: str) -> float:
        """
        Detect Sa frequency from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Detected Sa frequency in Hz
            
        Raises:
            RuntimeError: If Sa detection fails
        """
        print(f"Detecting tonic Sa from {audio_path}...")
        
        result = self.tonic_detector.ensemble_detection(audio_path, verbose=True)
        
        if 'sa_frequency' not in result:
            raise RuntimeError(f"Sa detection failed: {result.get('error', 'Unknown error')}")
        
        self.sa_frequency = result['sa_frequency']
        
        # Initialize quantizer and validator with detected Sa
        self.swara_quantizer = SwaraQuantizer(self.sa_frequency)
        self.grammar_validator = RagaGrammarValidator(self.raga_name)
        
        print(f"Detected Sa: {self.sa_frequency:.2f} Hz")
        print(f"Initialized for raga: {self.raga_name}")
        
        return self.sa_frequency
    
    def detect_tonic_from_audio(self, y: np.ndarray) -> float:
        """
        Detect Sa frequency from audio array.
        
        Args:
            y: Audio samples
            
        Returns:
            Detected Sa frequency in Hz
        """
        # Save to temporary file for tonic detection
        import tempfile
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            try:
                sf.write(tmp.name, y, self.sr)
                return self.detect_tonic_from_file(tmp.name)
            finally:
                os.unlink(tmp.name)  # Clean up temp file
    
    def analyze_frame(self, audio_frame: np.ndarray, 
                     timestamp_ms: float) -> Optional[FrameResult]:
        """
        Analyze single audio frame for pitch and grammar validation.
        
        Args:
            audio_frame: Audio samples for this frame
            timestamp_ms: Timestamp of frame start
            
        Returns:
            FrameResult or None if analysis fails
        """
        if self.swara_quantizer is None or self.grammar_validator is None:
            return None
        
        try:
            # Extract pitch using pYIN (same as tonic_sa_detection.py)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_frame,
                fmin=80,       # Low Sa ≈ 130 Hz, allow margin
                fmax=500,      # High Sa ≈ 520 Hz, with harmonics  
                sr=self.sr,
                frame_length=1024,
                hop_length=256
            )
            
            # Get the most confident pitch estimate
            if len(f0) == 0 or not np.any(voiced_flag):
                return FrameResult(
                    timestamp_ms=timestamp_ms,
                    frequency_hz=0,
                    voiced=False,
                    voicing_prob=0,
                    swara_result=None,
                    validation_event=None
                )
            
            # Use median of voiced estimates for stability
            voiced_f0 = f0[voiced_flag]
            voiced_conf = voiced_probs[voiced_flag]
            
            if len(voiced_f0) == 0:
                return None
            
            # Weight by confidence and take median
            weights = voiced_conf / np.sum(voiced_conf)
            frequency = np.average(voiced_f0, weights=weights)
            avg_confidence = np.mean(voiced_conf)
            
            # Quantize to swara
            swara_result = self.swara_quantizer.to_swara(frequency)
            if swara_result is None:
                return FrameResult(
                    timestamp_ms=timestamp_ms,
                    frequency_hz=frequency,
                    voiced=True,
                    voicing_prob=avg_confidence,
                    swara_result=None,
                    validation_event=None
                )
            
            # Validate against raga grammar
            validation_event = self.grammar_validator.validate_swara(
                swara_result, timestamp_ms)
            validation_event.frequency_hz = frequency
            
            return FrameResult(
                timestamp_ms=timestamp_ms,
                frequency_hz=frequency,
                voiced=True,
                voicing_prob=avg_confidence,
                swara_result=swara_result,
                validation_event=validation_event
            )
            
        except Exception as e:
            print(f"Frame analysis error: {e}")
            return None
    
    def analyze_file_streaming(self, audio_path: str, 
                              min_confidence: float = 0.3) -> Generator[FrameResult, None, None]:
        """
        Stream-analyze an audio file frame by frame.
        
        Args:
            audio_path: Path to audio file
            min_confidence: Minimum voicing confidence to process
            
        Yields:
            FrameResult objects for each valid frame
        """
        # Detect tonic first
        self.detect_tonic_from_file(audio_path)
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        
        # Process frame by frame
        for frame_start in range(0, len(y) - self.frame_length, self.hop_length):
            frame_end = frame_start + self.frame_length
            audio_frame = y[frame_start:frame_end]
            
            timestamp_ms = (frame_start / sr) * 1000
            
            result = self.analyze_frame(audio_frame, timestamp_ms)
            if result and result.voicing_prob >= min_confidence:
                yield result
    
    def analyze_file(self, audio_path: str, verbose: bool = True) -> Dict:
        """
        Complete offline analysis of an audio file.
        
        Args:
            audio_path: Path to audio file
            verbose: Print progress information
            
        Returns:
            Complete analysis report with summary statistics
        """
        if verbose:
            print(f"\nAnalyzing {audio_path} for raga {self.raga_name}")
            print("=" * 60)
        
        start_time = time.time()
        
        # Collect all frame results
        frame_results = list(self.analyze_file_streaming(audio_path))
        
        analysis_time = time.time() - start_time
        
        if not frame_results:
            return {
                'success': False,
                'error': 'No valid frames detected',
                'analysis_time': analysis_time
            }
        
        # Extract validation events
        validation_events = [r.validation_event for r in frame_results 
                           if r.validation_event is not None]
        
        # Compile summary
        total_frames = len(frame_results)
        voiced_frames = len(validation_events)
        error_events = [e for e in validation_events if e.error_type is not None]
        
        # Get summary from grammar validator
        grammar_summary = self.grammar_validator.get_validation_summary()
        
        # Audio duration
        duration_sec = frame_results[-1].timestamp_ms / 1000 if frame_results else 0
        
        report = {
            'success': True,
            'audio_file': audio_path,
            'raga': self.raga_name,
            'detected_sa_hz': self.sa_frequency,
            'duration_sec': duration_sec,
            'analysis_time_sec': analysis_time,
            'total_frames': total_frames,
            'voiced_frames': voiced_frames,
            'total_errors': len(error_events),
            'error_rate': len(error_events) / max(voiced_frames, 1),
            'grammar_summary': grammar_summary,
            'error_events': error_events,
            'frame_results': frame_results  # Full detail for debugging
        }
        
        if verbose:
            print(f"\nAnalysis Summary:")
            print(f"  Duration: {duration_sec:.1f}s | Frames: {voiced_frames}/{total_frames}")
            print(f"  Sa frequency: {self.sa_frequency:.2f} Hz")
            print(f"  Errors detected: {len(error_events)} ({report['error_rate']:.1%})")
            print(f"  Analysis time: {analysis_time:.2f}s")
            
            if error_events:
                print(f"\nError Breakdown:")
                error_types = {}
                for event in error_events:
                    et = event.error_type.value if event.error_type else 'unknown'
                    error_types[et] = error_types.get(et, 0) + 1
                
                for error_type, count in error_types.items():
                    print(f"  {error_type}: {count}")
        
        return report
    
    def reset_analysis(self):
        """Reset all analysis state for new file"""
        if self.grammar_validator:
            self.grammar_validator.reset()
        self.sa_frequency = None
        self.swara_quantizer = None

class BatchAnalyzer:
    """Analyze multiple audio files for comparative validation studies"""
    
    def __init__(self, raga_name: str):
        self.raga_name = raga_name
        self.pipeline = RealTimeGrammarPipeline(raga_name)
    
    def analyze_directory(self, dir_path: str, file_pattern: str = "*.wav") -> List[Dict]:
        """
        Analyze all matching files in a directory.
        
        Args:
            dir_path: Directory containing audio files
            file_pattern: Glob pattern for file matching
            
        Returns:
            List of analysis reports
        """
        import glob
        
        audio_files = glob.glob(os.path.join(dir_path, file_pattern))
        results = []
        
        for audio_file in audio_files:
            print(f"\nProcessing: {os.path.basename(audio_file)}")
            try:
                self.pipeline.reset_analysis()
                result = self.pipeline.analyze_file(audio_file, verbose=False)
                result['filename'] = os.path.basename(audio_file)
                results.append(result)
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                results.append({
                    'filename': os.path.basename(audio_file),
                    'success': False,
                    'error': str(e)
                })
        
        return results