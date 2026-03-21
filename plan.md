# Plan: Rule-Based Raga Grammar Validation System

## Context
- Capstone project: Automated Music Teacher for Carnatic music
- Existing: `tonic_sa_detection.py` (Sa detection via Carnatic HPS + pYIN ensemble)
- Goal: Grammar engine that takes pYIN pitch output → quantize swara → validate against raga rules
- Scope: 34 user-specified common ragas + all 72 Melakarta ragas (programmatic generation)
- Vakra ragas: YES (Kambhoji, Begada, Atana, Srikanti, etc.)
- Validation levels: (1) Forbidden swara detection, (2) Arohana/Avarohana sequence order

## Recommended Approach: Layered Hierarchical Rule Engine
Three layers — each layer checks different grammar depth:
- Layer 1: Immediate — forbidden (varja) swara set lookup → O(1)
- Layer 2: Sequential — FSM over Arohana/Avarohana sequence with direction detection
- Layer 3: (Future) phrase-level pattern matching

## Architecture

### Files
```
raga_grammar/
├── __init__.py
├── raga_database.py          # RAGA_DB dict: all 34 ragas + 72 melakarthas
├── swara_quantizer.py        # Hz → swara name using Sa-relative cents
├── grammar_validator.py      # Core FSM + rule engine
├── pitch_pipeline.py         # pYIN integration + direction detection
└── feedback_generator.py     # Human-readable error messages in English/Kannada
```

## Steps

### Phase 1: Raga Knowledge Base (raga_database.py)
1. Define SWARA_RATIOS dict — 12 positions with cent values (from tonic_sa_detection.py's CARNATIC_RATIOS)
2. For each of 34 ragas, encode: arohana (list), avarohana (list), varja_arohana (set), varja_avarohana (set), parent_mela (int), is_vakra_arohana (bool), is_vakra_avarohana (bool)
3. Programmatically generate all 72 Melakarta ragas using Katapayadi scheme (mela number → Ri/Ga/Ma/Dha/Ni variant selection)

**Vakra encoding**: vakra swaras encoded inline in the arohana/avarohana list itself:
- Kambhoji arohana: ["Sa","Ri2","Ga2","Ma1","Pa","Dha2","Sa"] (Ni absent — varja)
- Kambhoji avarohana: ["Sa","Ni2","Dha2","Pa","Ma1","Ga2","Ri2","Sa"] (Ni2 present only in descent)

### Phase 2: Swara Quantizer (swara_quantizer.py)
4. Map Sa-relative cents to 12 swara labels with ±25 cents tolerance window
5. SwaraQuantizer(sa_freq) class with .to_swara(f0_hz) → (swara_name, octave, cents_deviation) tuple
6. Handle octave folding: pitch in any octave maps to the same swara

### Phase 3: Grammar Validator (grammar_validator.py)
7. RagaGrammarFSM class per raga:
   - arohana_sequence / avarohana_sequence as ordered lists
   - current_direction ("ascending" / "descending" / "neutral")
   - current_position: index into current sequence
8. Direction detection: maintain last 3 quantized swaras, detect if moving up or down by swara ordinal value
9. Forbidden note check: instant O(1) — swara in raga's varja set for current direction → immediate FORBIDDEN_NOTE error
10. Sequence order check: does current swara follow the FSM's current position in arohana/avarohana? 
    - Allowed transitions: current swara == expected swara, OR skip forward (elision in gamaka)
    - Violation: swara appears much earlier than expected, or forbidden swara appears in wrong direction
11. On direction reversal: reset FSM position to start of new direction sequence

### Phase 4: Pitch Pipeline (pitch_pipeline.py)
12. RealTimeGrammarPipeline class:
    - Wraps TonicSaDetector from tonic_sa_detection.py
    - Runs pYIN on audio frames (2048 sample blocks, hop=512, ~23ms per frame at 22050 Hz)
    - Feeds quantized swaras to GrammarValidator in a streaming fashion
    - Returns list of ValidationEvent objects: {swara, timestamp, error_type, description}
13. Offline mode: analyze_file(path, raga_name) → full report

### Phase 5: Feedback Generator (feedback_generator.py)
14. Map error types to human-readable messages:
    - FORBIDDEN_NOTE: "Note {X} is varja (forbidden) in {raga} during ascent/descent"
    - SEQUENCE_VIOLATION: "Note {X} breaks {raga}'s Arohana — expected {Y}"
    - WRONG_DIRECTION: "Note {X} sung ascending but only allowed in descent"

## Relevant files
- `tonic_sa_detection.py` — reuse TonicSaDetector, CARNATIC_RATIOS, load_audio(), detect_by_pitch_histogram()
- `raga_grammar/raga_database.py` — new, most data-intensive file
- `raga_grammar/grammar_validator.py` — core logic: FSM + direction + forbidden check

## Verification
1. Unit test: validate known error sequences against Shankarabharanam (Ma2 is forbidden — should fire FORBIDDEN_NOTE)
2. Unit test: Kambhoji ascending — Ni should be flagged as varja
3. Unit test: Kalyani ascending — Ma2 is allowed, Ma1 is forbidden
4. Integration test: run pitch_pipeline on a sample Carnatic recording, check that no false errors fire on correct playing
5. Offline test: use the existing test audio files in drive-download folder as integration inputs

## Decisions
- Vakra encoding: inline in the sequence list (not separate zigzag field) — simplest to validate with FSM
- 72 Melakarta generation: programmatic from mela number (not all 72 hand-encoded)
- Tolerance: ±25 cents per swara (half of the smallest swara interval ~50 cents for enharmonics)
- Start with 34 listed ragas; Melakarta generation is a bonus pass
- No gamaka detection in this phase — sustained notes and gamaka-center note both quantize to same swara
