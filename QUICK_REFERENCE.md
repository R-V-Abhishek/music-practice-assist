# Quick Reference: Bulletproof Tonic-Ratio Pitch Detection

## The Algorithm (3 Lines)

1. **Detect Sa first** (1.5 seconds of audio) → Get tonic frequency
2. **Calculate 12 swara frequencies** from Sa using Carnatic ratios
3. **Match singer's frequency** against all 12 using ±35 cent tolerance

## Core Formula

```
Swara_Frequency_Hz = Sa_Frequency_Hz × Carnatic_Ratio

Deviation_Cents = 1200 × log₂(Singer_Frequency / Swara_Frequency)

Confidence = max(0, 1 - |Deviation_Cents| / 35)
```

## The 12 Carnatic Swaras

| Index | Name | Ratio | Example (Sa=260 Hz) |
|-------|------|-------|---------------------|
| 0 | Sa | 1.0 | 260.00 Hz |
| 1 | Ri1 | 256/243 | 273.91 Hz |
| 2 | Ri2 | 9/8 | 292.50 Hz |
| 3 | Ga1 | 32/27 | 308.15 Hz |
| 4 | Ga2 | 5/4 | 325.00 Hz |
| 5 | Ma1 | 4/3 | 346.67 Hz |
| 6 | Ma2 | 45/32 | 365.62 Hz |
| 7 | Pa | 3/2 | 390.00 Hz |
| 8 | Dha1 | 128/81 | 410.86 Hz |
| 9 | Dha2 | 5/3 | 433.33 Hz |
| 10 | Ni1 | 16/9 | 462.22 Hz |
| 11 | Ni2 | 15/8 | 487.50 Hz |

## Processing Pipeline

```
Microphone Audio (continuously streaming)
    ↓ [Buffer 768 samples = 35 ms]
    ↓
RMS Gate (loudness > 0.012)
    ↓ [Reject silence]
    ↓
YIN Pitch Detector (search range: 0.9×Sa to 2.1×Sa)
    ↓ [Extract raw pitch, only in tonic-focused range]
    ↓
Temporal Smoothing:
  - Median filter (9 frames)
  - EMA with α=0.25
  - Octave jump correction
  - Jump clamping (±45 cents/frame)
    ↓ [Stabilized frequency]
    ↓
SwaraQuantizer.to_swara_tonic_band(frequency)
    ↓ [Match against all 12 swaras]
    ↓
Swara Match (or None if outside ±35 cents)
    ↓
RagaGrammarValidator (check if swara is allowed)
    ↓
Output: SwaraResult {swara, octave, confidence, deviation}
```

## Python API

### Quick Start

```python
from raga_grammar.swara_quantizer import SwaraQuantizer

# Initialize with detected Sa frequency
sa = 260.0  # Hz (or from TonicSaDetector)
quantizer = SwaraQuantizer(sa_frequency=sa)

# Match a frequency
result = quantizer.to_swara_tonic_band(390.0)

if result:
    print(f"Swara: {result.swara}")              # "Pa"
    print(f"Confidence: {result.confidence}")    # 1.0 (100%)
    print(f"Deviation: {result.cents_deviation}") # 0.0 cents
    print(f"Octave: {result.octave}")            # 0
```

### Full Pipeline

```python
from raga_grammar.pitch_pipeline import RealTimeGrammarPipeline
import numpy as np

# Initialize for a raga
pipeline = RealTimeGrammarPipeline(raga_name="Bhairavi")

# Lock tonic Sa
pipeline.initialize_with_sa(sa_frequency=260.0)

# Process audio frame by frame
audio_frame = np.random.randn(768).astype(np.float32)  # 35 ms at 22050 Hz
result = pipeline.analyze_frame(audio_frame, timestamp_ms=0)

if result and result.swara_result:
    print(f"Detected: {result.swara_result.swara}")
```

### Live Processor

```python
from raga_grammar.live_audio_processor import LiveAudioProcessor, LiveProcessorConfig

config = LiveProcessorConfig(
    bootstrap_seconds=1.5,      # Sa detection time
    min_frame_rms=0.012,        # Loudness gate
    min_swara_confidence=0.45   # Confidence filter
)

processor = LiveAudioProcessor(raga_name="Bhairavi", config=config)

# Feed audio chunks (each ~256 samples from microphone)
events = processor.process_audio_chunk(
    chunk=audio_data,
    chunk_sample_rate=22050
)

for event in events:
    if event['type'] == 'frame_event':
        print(f"Swara: {event.get('swara')}")
```

## Key Configuration Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `TOLERANCE_CENTS` | 35 | 20-50 | How strict swara matching is |
| `min_frame_rms` | 0.012 | 0.005-0.02 | Silence/noise rejection threshold |
| `min_swara_confidence` | 0.45 | 0.3-0.7 | Filter weak matches |
| `_ema_alpha` | 0.25 | 0.15-0.4 | Smoothing responsiveness |
| `bootstrap_seconds` | 1.5 | 0.5-3.0 | Time to detect Sa |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| False positives (wrong swaras detected) | Increase `TOLERANCE_CENTS` or `min_swara_confidence` |
| Missing notes (valid notes not detected) | Decrease `TOLERANCE_CENTS` or `min_frame_rms` |
| Laggy detection | Lower `_ema_alpha` (0.15-0.20) for faster response |
| Jittery output | Raise `_ema_alpha` (0.30-0.35) for more smoothing |
| Silent phase causes crashes | Check `min_frame_rms` threshold |

## When to Use Which Method

| Use Case | Method |
|----------|--------|
| One-time file analysis | `RealTimeGrammarPipeline.analyze_file()` |
| Live practice session | `LiveAudioProcessor` |
| Custom frequency matching | `SwaraQuantizer.to_swara_tonic_band()` |
| Low-latency detection | Direct `analyze_frame()` calls |

## Verification Checklist

- [ ] Sa is detected correctly from bootstrap audio
- [ ] All 12 swaras can be matched within tolerance
- [ ] Octave folding works for high/low frequencies
- [ ] Confidence scores are high (>0.8) for in-tune notes
- [ ] Out-of-tune notes are rejected (return None)
- [ ] Temporal smoothing reduces frame jitter
- [ ] Grammar validation catches forbidden notes

## Important Notes

🔒 **Once Sa is locked, the system is deterministic** - Same frequency always maps to the same swara

📊 **Confidence indicates how close to the swara** - Use for filtering weak detections

🎵 **±35 cents tolerance** - Accepts natural vibrato (±5%) and gamakas

🔊 **RMS gate filters silence** - Prevents false detections in quiet passages

⏱️ **Processing latency ~35 ms per frame** - Fast enough for real-time feedback

## Files Modified

- `raga_grammar/swara_quantizer.py` - Bulletproof octave folding + tolerance checking
- `raga_grammar/pitch_pipeline.py` - Enhanced documentation and error handling
- `TONIC_RATIO_PITCH_DETECTION.md` - Complete algorithm documentation
