# Music Practice Assist — Master Implementation Plan

---

## Phase 0: Fix Current FSM Pipeline (DO FIRST)

> [!IMPORTANT]
> The FSM pipeline has 4 bugs that must be fixed **before** any DL work begins. Until these are fixed, the system gives wrong feedback for most ragas.

### Bug Summary

| # | File | Bug | Effect |
|---|------|-----|--------|
| 1 | `raga_database.py` | `SWARA_CENTS` has wrong cent values for `Ga1`, `Ga3`, `Ni1`, `Ni3` | Vakra detection broken for all ragas using these swaras |
| 2 | `swara_quantizer.py` | Missing 4 enharmonic alias names (`Ga1`, `Ri3`, `Ni1`, `Dha3`) | Sung notes map to wrong swara name → membership check fires incorrectly |
| 3 | `grammar_validator.py` | No enharmonic alias resolution after quantization | Quantizer returns `Ri2`, raga checks for `Ga1` → no match → no flag |
| 4 | `live_audio_processor.py` | `_build_alert_events` only handles `FORBIDDEN_NOTE`, drops `FORBIDDEN_PHRASE` | Vakra skip violations never reach the UI |

---

### Correct Carnatic Enharmonic Theory

**12 distinct pitches, 16 names.** Enharmonic pairs (same Hz, different names):

| Canonical (quantizer) | Alias (used in raga DB) | Cents |
|---|---|---|
| `Ri2` | `Ga1` | 204 |
| `Ri3` / `Ga2` | (both used, both canonical) | 294 |
| `Dha2` | `Ni1` | ~906 |
| `Ni2` | `Dha3` | ~996 |

Standalone (no enharmonic partner): `Ri1`, `Ga3`, `Ma1`, `Ma2`, `Pa`, `Dha1`, `Ni3`, `Sa`

---

### Fix 1 — `raga_database.py`: Correct `SWARA_CENTS`

#### [MODIFY] [raga_database.py](file:///d:/Coding/CAPSTONE/raga_grammar/raga_database.py)

```python
# BEFORE (wrong):
SWARA_CENTS = {
    'Sa': 0,    'Ri1': 90,  'Ri2': 204,  'Ri3': 294,
    'Ga1': 294, 'Ga2': 386, 'Ga3': 498,  'Ma1': 498,
    'Ma2': 612, 'Pa': 702,  'Dha1': 792, 'Dha2': 906,
    'Dha3': 996,'Ni1': 996, 'Ni2': 1088, 'Ni3': 1200   ← Ni3=1200 is next Sa!
}

# AFTER (correct):
SWARA_CENTS = {
    'Sa': 0,
    'Ri1': 90,
    'Ri2': 204,  'Ga1': 204,   # enharmonic pair
    'Ri3': 294,  'Ga2': 294,   # enharmonic pair
    'Ga3': 386,                # standalone — NOT same as Ma1
    'Ma1': 498,                # standalone
    'Ma2': 612,
    'Pa': 702,
    'Dha1': 792,
    'Dha2': 906, 'Ni1': 906,   # enharmonic pair
    'Dha3': 996, 'Ni2': 996,   # enharmonic pair
    'Ni3': 1088,               # standalone (Kakali Ni)
}
```

Also fix typos `"Da2"` → `"Dha2"` in `Śrīranjani`, `Kāpi`, `Kāmās`, `Nāṭakurinji` raga definitions.

---

### Fix 2 — `swara_quantizer.py`: Add Enharmonic Alias Names

#### [MODIFY] [swara_quantizer.py](file:///d:/Coding/CAPSTONE/raga_grammar/swara_quantizer.py)

The quantizer is raga-agnostic: it maps Hz → one canonical name per pitch. The 4 missing aliases need entries pointing to the **same ratio** as their canonical partner:

```python
CARNATIC_RATIOS = {
    'Sa': 1.0,      'Ri1': 256/243,
    'Ri2': 9/8,     'Ga1': 9/8,      # Ga1 = Ri2 (same ratio)
    'Ga2': 32/27,   'Ri3': 32/27,    # Ri3 = Ga2 (same ratio)
    'Ga3': 5/4,     'Ma1': 4/3,      # standalone — different ratios!
    'Ma2': 45/32,   'Pa': 3/2,
    'Dha1': 128/81,
    'Dha2': 5/3,    'Ni1': 5/3,      # Ni1 = Dha2 (same ratio)
    'Ni2': 16/9,    'Dha3': 16/9,    # Dha3 = Ni2 (same ratio)
    'Ni3': 15/8,
}
```

> [!IMPORTANT]
> Since duplicates exist in `CARNATIC_RATIOS`, the `to_swara_tonic_band()` method iterates all entries and picks the **closest**. When two entries have identical distances (enharmonics), it returns whichever dict iteration hits first. The validator resolves the correct name for the raga — see Fix 3.

---

### Fix 3 — `grammar_validator.py`: Enharmonic Alias Resolution

#### [MODIFY] [grammar_validator.py](file:///d:/Coding/CAPSTONE/raga_grammar/grammar_validator.py)

Add resolver and call it in `validate_swara()` immediately before the FSM check:

```python
# Module-level constant
ENHARMONIC_PAIRS = [
    ('Ri2', 'Ga1'),    # 204 cents
    ('Ri3', 'Ga2'),    # 294 cents
    ('Dha2', 'Ni1'),   # 906 cents
    ('Ni2', 'Dha3'),   # 996 cents
]

def _resolve_enharmonic(swara: str, raga_info: RagaInfo) -> str:
    """Return the enharmonic alias used by this raga, if applicable."""
    all_raga = set(raga_info.arohana) | set(raga_info.avarohana)
    for a, b in ENHARMONIC_PAIRS:
        if swara == a and b in all_raga and a not in all_raga:
            return b
        if swara == b and a in all_raga and b not in all_raga:
            return a
    return swara
```

In `RagaGrammarValidator.validate_swara()`, after getting `swara = swara_result.swara`, add:
```python
swara = _resolve_enharmonic(swara, self.raga_info)
```

---

### Fix 4 — `live_audio_processor.py`: Surface `FORBIDDEN_PHRASE` Alerts

#### [MODIFY] [live_audio_processor.py](file:///d:/Coding/CAPSTONE/raga_grammar/live_audio_processor.py)

```python
# BEFORE (line 190):
if validation is None or validation.error_type != ErrorType.FORBIDDEN_NOTE:

# AFTER:
_ALERT_ERRORS = {ErrorType.FORBIDDEN_NOTE, ErrorType.FORBIDDEN_PHRASE}
if validation is None or validation.error_type not in _ALERT_ERRORS:
```

Also update the emitted alert dict to use the actual error type:
```python
"alert": validation.error_type.value,   # 'forbidden_note' or 'forbidden_phrase'
```

---

### Phase 0 Verification

```bash
# Check corrected cents table
python -c "from raga_grammar.raga_database import SWARA_CENTS; [print(f'{k}: {v}') for k,v in sorted(SWARA_CENTS.items(), key=lambda x: x[1])]"

# Confirm enharmonic resolution works
python -c "
from raga_grammar.grammar_validator import _resolve_enharmonic, ENHARMONIC_PAIRS
from raga_grammar.raga_database import get_raga_info
raga = get_raga_info('Ānandabhairavi')
print(_resolve_enharmonic('Ri2', raga))   # → Ga2 (used in arohana)
print(_resolve_enharmonic('Ni2', raga))   # → Ni2 (raga uses Ni2 directly)
"

# Run existing test suite — must still pass
python -m pytest test_raga_grammar.py -v
```

**Manual test in live dashboard:**
1. Select **Ānandabhairavi** → sing `Pa Dha2 Ni2 Sa` ascending → alert fires for `Ni2`
2. Select **Varāḷi** → sing `Ga1 → Ma2` (skip `Ri1`) → alert fires as `FORBIDDEN_PHRASE`

---
---

## Phase 1–4: DL-Based Gamaka Analysis (after Phase 0 complete)

### Problem & Background

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
