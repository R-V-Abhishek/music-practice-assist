# DL-Based Gamaka Analysis — Implementation Plan

## Problem & Background

### Current FSM Pipeline (what exists today)

```
Audio Chunk → pYIN Pitch → SwaraQuantizer → RagaGrammarFSM → ValidationEvent → FeedbackGenerator → UI Alert
```

**What the FSM does well:**
- Forbidden note detection: O(1) set lookup (`varja_arohana` / `varja_avarohana`)
- Vakra skip detection: `ForbiddenPhraseDetector` on `PhraseBuffer`
- Characteristic phrase recognition: `CharacteristicPhraseDetector`
- Direction-aware validation: octave-aware pairwise `_get_step_direction()`
- Debounced real-time alerts: 500ms trigger / 300ms clear (`_build_alert_events`)

**What the FSM cannot do:**
- Gamakas are pitch contours, not discrete swaras the FSM sees them wrong
- Ri1 in Todi is a sustained oscillation ±50 cents — the quantizer collapses this to a flat `Ri1`
- No score of *how well* a gamaka is rendered vs. a master
- No "you sang Ri1 correctly in pitch but without the required kampita"

### Why a fixed 1-2 min window is wrong

| Issue | Why |
|---|---|
| Arbitrary boundary | Phrase end ≠ wall clock tick |
| Bad UX | Mistake at 0:05, reinforced for 2 min before correction |
| No temporal alignment | "Weak kampita" — *which* Ga? *When*? |
| Memory waste | Raw audio at 22 kHz × 120s ≈ 5.3M samples/session |

**Solution:** Phrase-boundary-driven DL inference, glued on top of the running FSM loop.

---

## Architecture: Tiered Feedback (FSM + DL)

```
                    RAW AUDIO STREAM
                           |
                    PITCH EXTRACTION (pYIN — unchanged)
                    → Hz → cents relative to tonic
                           |
          EXISTING PER-FRAME FSM LOOP (unchanged)
          SwaraQuantizer → RagaGrammarFSM → ValidationEvent
          → _build_alert_events() → UI forbidden-note alert
                           |
                    pitch contour ring buffer (no raw audio)
                           |
                    PHRASE SEGMENTOR (new module)
                    Watches for phrase boundaries:
                    - silence > 300ms (RMS below threshold)
                    - nyasa swara held > 800ms
                    - energy valley between melodic runs
                    Emits: Phrase(start_ms, end_ms, contour)
                           |
         only fires after phrase boundary detected
              |                              |
     Tier 2+ raga                    Tier 3+ raga (session end)
              |                              |
    GAMAKA CLASSIFIER              QUALITY SCORER (Siamese)
    TDNN+BiLSTM (Model 1)          Learner vs master embedding
    kampita/jaru/nokku/plain       → score 0-1 + sub-scores
    + kampita absence check        oscillation/landing/shape
    (rule, Model 3)
              |                              |
    phrase-level hint              session quality report
    "kampita on Ga2 too slow"      "Ri1 kampita: 0.72/1.0"
```

### Tiered Feedback by Raga Difficulty

| Tier | Ragas | Active layers |
|---|---|---|
| 1 — Basic | Mohanam, Madhyamavati, Bilahari | FSM only (no DL inference) |
| 2 — Intermediate | Shankarabharanam, Kalyani, Kambhoji | FSM + phrase-end gamaka hint |
| 3 — Advanced | Todi, Varali, Bhairavi, Anandabhairavi | FSM + DL classifier + absence check |
| 4 — Expert | Todi, Varali (full rendition) | All tiers + session quality score |

Tag each raga with `difficulty_tier: int` (1–4) and `gamaka_intensity: float` (0–1) in `RagaInfo`.

---

## Deployment Constraint: Mobile Edge

> [!IMPORTANT]
> Target: iOS/Android edge inference. No server round-trips during practice.

| Constraint | Decision |
|---|---|
| No GPU on device | CPU/NPU only. Max ~10ms per phrase on mid-range phone |
| Model size | Total DL footprint < 5 MB combined |
| Export | PyTorch → ONNX → TFLite (Android) / CoreML (iOS) |
| Quantization | INT8 post-training quantization mandatory |
| Pitch extraction | pYIN (already in use, CPU-efficient). No CREPE. |
| Buffer | Pitch contour ring buffer in RAM — not raw audio |

Estimated: TDNN+BiLSTM at hidden=64 ≈ 650 KB quantized. Siamese head shares encoder → +100 KB. Total < 2 MB.

---

## Model Design

### Gamaka Taxonomy: 4 Classes

| Class | Acoustic signature |
|---|---|
| **kampita** | Sinusoidal pitch modulation ±30–70 cents @ 4–8 Hz |
| **jaru** | Monotone ramp ending at target pitch (up or down) |
| **nokku** | Sharp attack spike + exponential decay to target |
| **plain** | Flat pitch contour near target ± small noise |

Jaru direction (up/down) = secondary attribute on `jaru`, not a separate class.

### Model 1: Gamaka Classifier (TDNN + BiLSTM)

**Input:** pitch contour in cents relative to tonic, variable length (50–300 frames @ 10ms hop)
**Output:** class (4) + confidence + jaru direction

```
pitch contour (T × 1)
    |
Feature extraction (fixed, no learned params)
    → delta pitch (velocity)
    → delta-delta pitch (acceleration)
    → local oscillation freq (FFT over 50ms window)
    (T × 3)
    |
TDNN Block — 2 layers
    Layer 1: context [-1, 0, +1] → filters=32   (kampita at 100ms scale)
    Layer 2: context [-3, 0, +3] → filters=64   (kampita at 300ms scale)
    |
BiLSTM — 1 layer, hidden=64
    |
Mean pooling over time
    |
FC(64→32) → ReLU → FC(32→4) → Softmax
```

Training: PyTorch → INT8 quantize → ONNX → TFLite/CoreML
Size: ~650 KB quantized | Latency: ~3–6ms/phrase on Snapdragon 778G class

### Model 2: Quality Scorer (Siamese BiLSTM)

**Input:** learner phrase contour + master exemplar contour (same gamaka type, same swara)
**Output:** quality score (0–1) + sub-scores

```
Learner contour → Shared encoder (Model 1 frozen) → Embedding (64-d)
                                                           |
                                                   Cosine sim + L2 dist
                                                           |
Master contour  → Shared encoder                → Embedding (64-d)
                                                           |
                                                   FC → sub-scores:
                                                   - oscillation_match
                                                   - landing_accuracy
                                                   - shape_similarity
```

Trained with **triplet loss**.
Encoder frozen from Model 1 → Siamese head adds only ~100 KB.

### Model 3: Kampita Absence Detector (rule-based, no ML)

After Model 1 classifies a swara as `plain`, check `RagaInfo.swara_gamaka_map`:
- If swara has `requires_kampita = True` AND confidence > 0.7 → emit `ImprovementHint`
- Message: *"Try adding kampita on Ga2 — in Todi, this note is rarely sung straight"*
- Generalized: works for all ragas with a populated `swara_gamaka_map`

---

## Integration with Existing Code

### 1. `raga_database.py` — extend `RagaInfo`

#### [MODIFY] [raga_database.py](file:///d:/Coding/CAPSTONE/raga_grammar/raga_database.py)

Add to `RagaInfo` dataclass (non-breaking — all new fields have defaults):

```python
difficulty_tier: int = 1           # 1=basic, 2=intermediate, 3=advanced, 4=expert
gamaka_intensity: float = 0.0      # 0=none, 1=pervasive
swara_gamaka_map: Dict[str, Dict] = field(default_factory=dict)
# {swara: {"expected": ["kampita"], "requires_kampita": True}}
```

Example for Todi in `_RAGA_DEFINITIONS`:
```python
"difficulty_tier": 3,
"gamaka_intensity": 0.9,
"swara_gamaka_map": {
    "Ri1": {"expected": ["kampita"], "requires_kampita": True},
    "Ga2": {"expected": ["kampita", "nokku"], "requires_kampita": True},
    "Dha1": {"expected": ["kampita"], "requires_kampita": False},
    "Ni2": {"expected": ["jaru", "plain"], "requires_kampita": False}
}
```

Pass these through `_build_raga_info()` into the `RagaInfo` constructor.

---

### 2. New module: `raga_grammar/phrase_segmentor.py`

#### [NEW] [phrase_segmentor.py](file:///d:/Coding/CAPSTONE/raga_grammar/phrase_segmentor.py)

```python
@dataclass
class Phrase:
    start_ms: float
    end_ms: float
    pitch_contour: np.ndarray   # cents relative to tonic, shape (T,)
    dominant_swara: Optional[str]

class PhraseSegmentor:
    """Watches the pitch contour ring buffer from LiveAudioProcessor.
    Emits Phrase objects when boundaries are detected."""

    def push_frame(self, pitch_cents: float, rms: float, ts_ms: float) -> Optional[Phrase]:
        # Returns a Phrase when a boundary is crossed, else None
        ...
```

Boundary detection:
- `rms < silence_threshold` for > 300ms → phrase end
- Same swara sustained > 800ms (nyasa) → phrase end
- Energy valley between two melodic runs (local RMS minimum) → phrase end

---

### 3. New module: `raga_grammar/gamaka_classifier.py`

#### [NEW] [gamaka_classifier.py](file:///d:/Coding/CAPSTONE/raga_grammar/gamaka_classifier.py)

```python
@dataclass
class GamakaResult:
    gamaka_type: str        # "kampita" | "jaru" | "nokku" | "plain"
    confidence: float
    jaru_direction: Optional[str]   # "up" | "down" | None
    dominant_swara: Optional[str]

class GamakaClassifier:
    def __init__(self, model_path: str): ...
    def classify_phrase(self, pitch_contour: np.ndarray) -> GamakaResult: ...
```

---

### 4. New module: `raga_grammar/quality_scorer.py`

#### [NEW] [quality_scorer.py](file:///d:/Coding/CAPSTONE/raga_grammar/quality_scorer.py)

```python
@dataclass
class QualityScore:
    overall: float
    oscillation_match: float
    landing_accuracy: float
    shape_similarity: float

class QualityScorer:
    def __init__(self, model_path: str, exemplar_bank_path: str): ...
    def score_phrase(
        self, learner_contour: np.ndarray, raga_name: str, swara: str
    ) -> Optional[QualityScore]: ...
```

---

### 5. `live_audio_processor.py` — add phrase-gated DL loop

#### [MODIFY] [live_audio_processor.py](file:///d:/Coding/CAPSTONE/raga_grammar/live_audio_processor.py)

The existing per-frame FSM loop in `process_audio_chunk()` stays **completely unchanged**.

New in `__init__()`:
```python
raga_info = get_raga_info(raga_name)
self._phrase_segmentor = PhraseSegmentor()
self._gamaka_classifier = GamakaClassifier(model_path) if raga_info.difficulty_tier >= 2 else None
self._quality_scorer = QualityScorer(model_path, exemplar_path) if raga_info.difficulty_tier >= 3 else None
```

New path in `process_audio_chunk()` — runs **after** existing frame loop:
```python
phrase = self._phrase_segmentor.push_frame(pitch_cents, rms, ts_ms)
if phrase is not None and self._gamaka_classifier is not None:
    result = self._gamaka_classifier.classify_phrase(phrase.pitch_contour)
    events.extend(self._build_gamaka_events(phrase, result))
```

`_build_gamaka_events()` emits:
```json
{"type": "gamaka_hint", "gamaka": "kampita", "swara": "Ga2", "message": "..."}
```

---

### 6. `feedback_generator.py` — gamaka templates

#### [MODIFY] [feedback_generator.py](file:///d:/Coding/CAPSTONE/raga_grammar/feedback_generator.py)

```python
def generate_gamaka_feedback(
    self, result: GamakaResult, quality: Optional[QualityScore],
    swara: str, raga_name: str
) -> Dict:
    # "Your kampita on Ga2 is oscillating too slowly (4 Hz, expected ~6 Hz)"
    # "Jaru landing on Ri1 is sharp by 15 cents"
    # "Try adding kampita on Ga2 — in Todi this note is rarely sung straight"
    ...
```

---

## Data Strategy

1. **CompMusic dataset** (MTG Barcelona) — Carnatic recordings, some annotations
2. **Synthetic generation** — programmatic contours for pretraining:
   - kampita: sinusoidal, vary freq 4–8 Hz, amplitude 30–70 cents
   - jaru: linear/curved ramp, vary slope + offset
   - nokku: Gaussian spike + exponential decay
   - plain: flat ± small Gaussian noise
3. **Teacher annotation tool** — extend web dashboard (Phase 4)
4. **Self-supervised pretraining** — pitch autoencoder first, then fine-tune on labeled data

> [!IMPORTANT]
> Start with synthetic + CompMusic. Teacher annotation is Phase 4.

---

## Phased Execution

### Phase 1: Infrastructure (no ML)
- [ ] Add `difficulty_tier`, `gamaka_intensity`, `swara_gamaka_map` to all ragas in `raga_database.py`
  - Tier 3: Todi, Bhairavi, Anandabhairavi | Tier 4: Varali | Tier 2: Shankarabharanam, Kalyani, Kambhoji | Tier 1: Mohanam, Bilahari
- [ ] Build `phrase_segmentor.py` (energy/silence/nyasa detection, no ML)
- [ ] Wire `PhraseSegmentor` into `LiveAudioProcessor.process_audio_chunk()` alongside FSM loop
- [ ] Add gamaka feedback templates to `feedback_generator.py`
- [ ] Tiered mode selection: skip DL init if `difficulty_tier < 2`
- [ ] Unit tests for `PhraseSegmentor` on recorded Todi/Mohanam snippets

### Phase 2: Synthetic Data + Gamaka Classifier
- [ ] Build synthetic gamaka contour generator
- [ ] Build TDNN+BiLSTM gamaka classifier in PyTorch
- [ ] Train on synthetic, validate on CompMusic manually
- [ ] Export: PyTorch → ONNX → TFLite + CoreML (INT8)
- [ ] Build `gamaka_classifier.py` ONNX inference wrapper
- [ ] Integrate classifier + kampita absence check into phrase-gated loop
- [ ] Regression test: Mohanam/Kalyani FSM alerts must fire correctly with no DL interference

### Phase 3: Quality Scoring
- [ ] Curate master exemplar bank from CompMusic
- [ ] Train Siamese quality scorer (triplet loss, freeze Phase 2 encoder)
- [ ] Quantize and export Siamese head
- [ ] Build `quality_scorer.py`, wire into `get_session_summary()`
- [ ] Emit session-level sub-scores in summary dict

### Phase 4: Teacher Annotation Tool + Fine-tuning
- [ ] Extend web dashboard: phrase playback → gamaka label → submit
- [ ] Collect real labels → fine-tune Phase 2 model
- [ ] Re-export quantized models
- [ ] Iterative improvement loop

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| FSM loop untouched | Zero regression risk on basic/intermediate ragas |
| Pitch contour ring buffer, not raw audio | Avoids 5.3M sample memory per session |
| pYIN (already in use), not CREPE | CREPE too heavy for mobile edge |
| DL skipped for tier-1 ragas | No gamaka scoring needed for Mohanam, Bilahari |
| Encoder shared between classifier + quality scorer | Total model < 2 MB |
| Kampita absence as rule (Model 3), not ML | Avoids class imbalance; `swara_gamaka_map` encodes musical truth |
| 4-class taxonomy | Trainable with limited data; acoustically distinguishable |

---

## Verification Plan

- **PhraseSegmentor:** unit test on Todi + Mohanam recordings, check phrase count and boundary timing
- **Gamaka classifier:** accuracy on held-out synthetic + CompMusic (target > 85% on synthetic)
- **Quality scorer:** Spearman correlation between model scores and teacher rankings
- **Integration:** run live Todi session → gamaka hints fire correctly, FSM alerts still fire on wrong swaras, Mohanam/Kalyani see zero DL overhead
- **Export:** ONNX on Android emulator, verify < 10ms/phrase latency
