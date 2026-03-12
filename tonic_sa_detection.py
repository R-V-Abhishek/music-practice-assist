"""
Robust Tonic Sa Detection from Polyphonic Carnatic Music (v3)

Uses Carnatic-aware Harmonic Product Spectrum as the primary method,
with standard HPS and pitch histogram for validation.

Key insight: The tambura drone always sounds Sa and Pa (3:2 ratio).
By scoring each candidate Sa frequency using a weighted geometric mean
of spectral energy at Sa, Pa (1.5x), upper_Sa (2x), and higher harmonics,
the correct Sa is robustly identified even when the fundamental is weak.
"""

import numpy as np
import librosa
import scipy.signal as signal
from typing import Dict
import warnings

warnings.filterwarnings('ignore')


class TonicSaDetector:
    """
    Detects the tonic Sa frequency from polyphonic Carnatic music.
    
    Primary method: Carnatic HPS (weighted geometric mean at musical ratios)
    Validation: Standard HPS + Pitch class histogram from pYIN
    """
    
    SA_FREQUENCIES = {'low': 130, 'standard': 260, 'high': 520}
    
    # Carnatic note ratios from Sa
    CARNATIC_RATIOS = {
        'Sa': 1.0, 'Ri1': 256/243, 'Ri2': 9/8, 'Ga1': 32/27,
        'Ga2': 5/4, 'Ma1': 4/3, 'Ma2': 45/32, 'Pa': 3/2,
        'Dha1': 128/81, 'Dha2': 5/3, 'Ni1': 16/9, 'Ni2': 15/8,
    }
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self._audio_cache = {}
    
    def load_audio(self, audio_path: str) -> tuple:
        """Load and cache audio file (supports MP3, WAV, FLAC, OGG, etc.)."""
        if audio_path in self._audio_cache:
            return self._audio_cache[audio_path]
        try:
            y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
            self._audio_cache[audio_path] = (y, sr)
            return y, sr
        except Exception as e:
            raise RuntimeError(f"Failed to load '{audio_path}': {e}")
    
    def _get_median_spectrum(self, y, sr, n_fft=8192):
        """
        Compute median magnitude spectrum across all STFT frames.
        Median emphasizes persistent sounds (tambura drone) over transient melody.
        """
        S = librosa.stft(y, n_fft=n_fft, hop_length=n_fft // 4)
        mag = np.abs(S)
        median_mag = np.median(mag, axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        return freqs, median_mag
    
    def _score_carnatic_hps(self, spec, freq_res, sr, candidates):
        """Core Carnatic HPS scoring: weighted geometric mean in log domain."""
        ratios =  [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
        weights = [0.3, 1.0, 0.8, 0.5, 0.3, 0.2]
        log_scores = np.zeros(len(candidates))
        for ratio, w in zip(ratios, weights):
            targets = candidates * ratio
            bin_idx = np.round(targets / freq_res).astype(int)
            valid = (bin_idx < len(spec)) & (targets < sr / 2)
            bin_idx = np.clip(bin_idx, 0, len(spec) - 1)
            vals = spec[bin_idx].copy()
            vals[~valid] = 0.001
            vals[vals < 0.001] = 0.001
            log_scores += w * np.log(vals)
        return log_scores

    def detect_by_carnatic_hps(self, audio_path: str,
                                fmin: float = 80, fmax: float = 300,
                                step: float = 0.5) -> Dict:
        """
        Carnatic-aware Harmonic Product Spectrum (primary method).
        
        Scores each candidate Sa using weighted geometric mean of spectral
        energy at musically significant positions:
        - Sa (fundamental)
        - Pa = 1.5 x Sa (perfect fifth, from tambura)
        - Upper Sa = 2 x Sa (octave, from tambura)
        - 3rd, 4th, 5th harmonics
        
        The geometric mean (multiplicative) ensures ALL positions must have
        energy, making it robust against coincidental alignments.
        
        Returns the top result AND a list of strong candidates for
        cross-validation with other methods.
        """
        y, sr = self.load_audio(audio_path)
        
        # Coarse scan with n_fft=8192 (broader bins capture more energy)
        n_fft_coarse = 8192
        _, spec_c = self._get_median_spectrum(y, sr, n_fft_coarse)
        freq_res_c = sr / n_fft_coarse
        
        candidates = np.arange(fmin, fmax + step, step)
        log_scores = self._score_carnatic_hps(spec_c, freq_res_c, sr, candidates)
        scores = np.exp(log_scores)
        scores /= np.max(scores) + 1e-20
        
        # Collect strong candidates (score > 0.4 of max)
        strong_candidates = []
        for i in range(len(candidates)):
            if scores[i] > 0.4:
                strong_candidates.append((candidates[i], float(scores[i])))
        strong_candidates.sort(key=lambda x: -x[1])
        
        # Coarse best
        coarse_best = candidates[np.argmax(scores)]
        
        # Fine-tune with n_fft=16384 for higher precision, ±5 Hz, 0.1 Hz
        n_fft_fine = 16384
        _, spec_f = self._get_median_spectrum(y, sr, n_fft_fine)
        freq_res_f = sr / n_fft_fine
        
        fine_candidates = np.arange(
            max(fmin, coarse_best - 5),
            min(fmax, coarse_best + 5) + 0.1,
            0.1
        )
        fine_log = self._score_carnatic_hps(spec_f, freq_res_f, sr, fine_candidates)
        fine_scores = np.exp(fine_log)
        sa_freq = fine_candidates[np.argmax(fine_scores)]
        confidence = float(scores[np.argmax(scores)])
        
        return {
            'method': 'carnatic_hps',
            'sa_frequency': float(sa_freq),
            'confidence': confidence,
            '_strong_candidates': strong_candidates,
            '_spec_fine': spec_f,
            '_freq_res_fine': freq_res_f,
        }
    
    def detect_by_hps(self, audio_path: str, num_harmonics: int = 5) -> Dict:
        """
        Standard Harmonic Product Spectrum (validation method).
        Multiplies spectrum with integer-downsampled copies to find fundamental.
        Works well when the fundamental has detectable energy.
        """
        y, sr = self.load_audio(audio_path)
        
        n_fft = 16384
        freqs, spec = self._get_median_spectrum(y, sr, n_fft)
        
        # Build HPS by multiplying downsampled spectra
        min_len = len(spec)
        for h in range(2, num_harmonics + 1):
            min_len = min(min_len, len(spec) // h)
        
        hps = spec[:min_len].copy()
        for h in range(2, num_harmonics + 1):
            hps *= spec[::h][:min_len]
        
        hps /= np.max(hps) + 1e-20
        freqs_hps = freqs[:min_len]
        
        mask = (freqs_hps >= 80) & (freqs_hps <= 300)
        if not np.any(mask):
            return {'method': 'hps', 'sa_frequency': 130.0, 'confidence': 0.0}
        
        sa_freq = float(freqs_hps[mask][np.argmax(hps[mask])])
        confidence = float(np.max(hps[mask]))
        
        return {'method': 'hps', 'sa_frequency': sa_freq, 'confidence': confidence}
    
    def detect_by_pitch_histogram(self, audio_path: str) -> Dict:
        """
        Pitch class histogram from pYIN (validation method).
        Tracks pitch, folds into one octave, finds the most common note.
        The most frequent pitch class should be Sa (due to tambura drone).
        """
        y, sr = self.load_audio(audio_path)
        
        # Analyze first 60 seconds for speed
        max_samples = sr * 60
        y_short = y[:max_samples] if len(y) > max_samples else y
        
        try:
            f0, voiced, probs = librosa.pyin(
                y_short, fmin=80, fmax=500, sr=sr, frame_length=2048
            )
        except Exception:
            return {'method': 'pitch_histogram', 'sa_frequency': 130.0, 'confidence': 0.0}
        
        valid = voiced & (probs > 0.1)
        vf0 = f0[valid]
        vp = probs[valid]
        
        if len(vf0) < 20:
            return {'method': 'pitch_histogram', 'sa_frequency': 130.0, 'confidence': 0.0}
        
        # Fold into one octave using cents
        cents = 1200 * np.log2(vf0 + 1e-10)
        pc = cents % 1200
        
        hist, edges = np.histogram(pc, bins=600, range=(0, 1200), weights=vp)
        centers = (edges[:-1] + edges[1:]) / 2
        hist = np.convolve(hist, signal.windows.gaussian(21, 3), mode='same')
        
        peak_cent = centers[np.argmax(hist)]
        base = 2 ** (peak_cent / 1200)
        
        # Map to frequency in Sa range (80-300 Hz)
        candidates = [base * 2**n for n in range(6, 10) if 70 <= base * 2**n <= 350]
        
        if not candidates:
            return {'method': 'pitch_histogram', 'sa_frequency': float(np.median(vf0)),
                    'confidence': 0.1}
        
        # Pick candidate with most support from actual pitch data
        best_c, best_s = candidates[0], 0
        for c in candidates:
            s = np.sum(np.abs(vf0 - c) < 5) + 0.5 * np.sum(np.abs(vf0 - 2*c) < 5)
            if s > best_s:
                best_s, best_c = s, c
        
        confidence = min(float(best_s / len(vf0)) * 3, 1.0)
        
        return {
            'method': 'pitch_histogram',
            'sa_frequency': float(best_c),
            'confidence': confidence,
        }
    
    def _fine_tune_candidate(self, coarse_freq, spec_fine, freq_res_fine, sr,
                              fmin=80, fmax=300, window=5):
        """Fine-tune a coarse candidate using high-resolution spectrum."""
        fine_cands = np.arange(
            max(fmin, coarse_freq - window),
            min(fmax, coarse_freq + window) + 0.1,
            0.1
        )
        fine_log = self._score_carnatic_hps(spec_fine, freq_res_fine, sr, fine_cands)
        return float(fine_cands[np.argmax(fine_log)])

    def ensemble_detection(self, audio_path: str, verbose: bool = True) -> Dict:
        """
        Ensemble detection with cross-validation.
        
        Strategy:
        1. Carnatic HPS generates top candidate + strong secondary candidates
        2. Standard HPS provides an independent estimate
        3. If Carnatic HPS top matches Standard HPS → use it (high confidence)
        4. If not, check if any strong Carnatic HPS candidate matches
           Standard HPS → cross-validate (prevents dominant sung note from
           overriding the tambura drone detection)
        5. If no match at all → trust Carnatic HPS top (Standard HPS may be wrong)
        6. Pitch histogram used for additional consensus check
        """
        results = {}
        
        if verbose:
            print("Running detection methods...")
        
        # Primary: Carnatic HPS (returns strong candidates too)
        try:
            results['carnatic_hps'] = self.detect_by_carnatic_hps(audio_path)
            if verbose and 'sa_frequency' in results['carnatic_hps']:
                r = results['carnatic_hps']
                print(f"  Carnatic HPS: {r['sa_frequency']:.2f} Hz (conf: {r['confidence']:.3f})")
        except Exception as e:
            if verbose:
                print(f"  Carnatic HPS: Error - {e}")
        
        # Validation: Standard HPS
        try:
            results['hps'] = self.detect_by_hps(audio_path)
            if verbose and 'sa_frequency' in results['hps']:
                r = results['hps']
                print(f"  Standard HPS: {r['sa_frequency']:.2f} Hz (conf: {r['confidence']:.3f})")
        except Exception as e:
            if verbose:
                print(f"  Standard HPS: Error - {e}")
        
        # Validation: Pitch histogram
        try:
            results['pitch_histogram'] = self.detect_by_pitch_histogram(audio_path)
            if verbose and 'sa_frequency' in results['pitch_histogram']:
                r = results['pitch_histogram']
                print(f"  Pitch histogram: {r['sa_frequency']:.2f} Hz (conf: {r['confidence']:.3f})")
        except Exception as e:
            if verbose:
                print(f"  Pitch histogram: Error - {e}")
        
        # --- Cross-validation decision logic ---
        carnatic = results.get('carnatic_hps', {})
        hps_result = results.get('hps', {})
        
        if 'sa_frequency' not in carnatic:
            if 'sa_frequency' in hps_result:
                final_freq = hps_result['sa_frequency']
            else:
                return {'success': False, 'error': 'All detection methods failed'}
        else:
            carnatic_top = carnatic['sa_frequency']
            strong_cands = carnatic.get('_strong_candidates', [])
            spec_fine = carnatic.get('_spec_fine')
            freq_res_fine = carnatic.get('_freq_res_fine')
            y, sr = self.load_audio(audio_path)
            
            final_freq = carnatic_top  # Default: trust Carnatic HPS
            decision = 'carnatic_top'
            
            if 'sa_frequency' in hps_result:
                hps_freq = hps_result['sa_frequency']
                
                # Check if Carnatic HPS top already agrees with Standard HPS
                ratio_top = max(carnatic_top, hps_freq) / min(carnatic_top, hps_freq)
                
                if ratio_top < 1.10:
                    # They agree — high confidence, use Carnatic HPS top
                    final_freq = carnatic_top
                    decision = 'agreement'
                    if verbose:
                        print(f"  -> Carnatic HPS & Standard HPS agree")
                else:
                    # They disagree — check if any secondary Carnatic candidate
                    # matches Standard HPS (cross-validation)
                    best_xval, best_xval_score = None, 0
                    for cand_f, cand_s in strong_cands:
                        r = max(cand_f, hps_freq) / min(cand_f, hps_freq)
                        if r < 1.10 and cand_s > best_xval_score:
                            best_xval, best_xval_score = cand_f, cand_s
                    
                    if best_xval is not None and best_xval_score > 0.4:
                        # Cross-validated: a strong Carnatic candidate matches HPS
                        if spec_fine is not None:
                            final_freq = self._fine_tune_candidate(
                                best_xval, spec_fine, freq_res_fine, sr)
                        else:
                            final_freq = best_xval
                        decision = 'cross_validated'
                        if verbose:
                            print(f"  -> Cross-validated: Carnatic candidate "
                                  f"{best_xval:.1f} Hz matches Standard HPS "
                                  f"(score={best_xval_score:.3f})")
                    else:
                        # No match — trust Carnatic HPS top
                        if verbose:
                            print(f"  -> No HPS match; using Carnatic HPS top")
        
        # Additional consensus check with pitch histogram
        ph = results.get('pitch_histogram', {})
        if 'sa_frequency' in ph:
            ratio_ph = max(ph['sa_frequency'], final_freq) / min(ph['sa_frequency'], final_freq)
            if ratio_ph < 1.10:
                # Slight refinement via weighted average
                w_main = 0.85
                w_ph = 0.15 * ph.get('confidence', 0.5)
                final_freq = (final_freq * w_main + ph['sa_frequency'] * w_ph) / (w_main + w_ph)
                if verbose:
                    print(f"  -> pitch_histogram agrees ({ph['sa_frequency']:.2f} Hz)")
        
        if verbose:
            print(f"\n  => Detected Sa: {final_freq:.2f} Hz")
        
        # Clean internal keys from results before returning
        for key in list(results.get('carnatic_hps', {}).keys()):
            if key.startswith('_'):
                del results['carnatic_hps'][key]
        
        return {
            'method': 'ensemble',
            'sa_frequency': float(final_freq),
            'individual_results': results,
        }
    
    def get_nearest_standard_sa(self, detected_freq: float) -> Dict:
        """Find the nearest standard Carnatic Sa frequency."""
        dist = {o: abs(detected_freq - f) for o, f in self.SA_FREQUENCIES.items()}
        nearest = min(dist, key=dist.get)
        nf = self.SA_FREQUENCIES[nearest]
        return {
            'nearest_standard': nf,
            'octave': nearest,
            'distance_cents': 1200 * np.log2(detected_freq / nf),
            'difference_hz': detected_freq - nf,
        }


def example_usage(audio_file_path: str = None):
    """Interactive example usage with detailed output."""
    import time
    import os
    
    detector = TonicSaDetector()
    
    if audio_file_path is None:
        audio_file_path = input(
            "\nEnter the path to your audio file (MP3, WAV, FLAC, OGG, etc.): "
        ).strip()
    
    if not audio_file_path or not os.path.exists(audio_file_path):
        print(f"Error: File not found: {audio_file_path}")
        return
    
    print(f"\nAnalyzing: {audio_file_path}")
    print("=" * 60)
    print("TONIC SA DETECTION (Carnatic HPS + HPS + Pitch Histogram)")
    print("=" * 60)
    
    start_time = time.time()
    result = detector.ensemble_detection(audio_file_path, verbose=True)
    elapsed = time.time() - start_time
    
    if 'sa_frequency' in result:
        sa = result['sa_frequency']
        print(f"\n{'=' * 60}")
        print(f"DETECTED Sa: {sa:.2f} Hz")
        print(f"Processing time: {elapsed:.2f} seconds")
        
        std = detector.get_nearest_standard_sa(sa)
        print(f"\nNearest Standard Sa: {std['nearest_standard']} Hz ({std['octave']})")
        print(f"Difference: {std['difference_hz']:.2f} Hz ({std['distance_cents']:.1f} cents)")
        
        print(f"\n{'=' * 60}")
        print("METHOD BREAKDOWN")
        print("=" * 60)
        for method, res in result.get('individual_results', {}).items():
            if 'sa_frequency' in res:
                print(f"  {method}: {res['sa_frequency']:.2f} Hz "
                      f"(confidence: {res.get('confidence', 0):.3f})")
        print(f"\nMethods in consensus: {result.get('consensus_count', 1)}")
    else:
        print(f"\nDetection failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    example_usage()
