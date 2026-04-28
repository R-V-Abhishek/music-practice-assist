# Shruti Monitor: Note Detection Implementation (Deep Dive)

This document explains exactly how the app detects pitch and maps it to swaras/notes, based on the current implementation in:

- `app.js` (main-thread detection, smoothing, UI mapping)
- `pitch-processor.js` (AudioWorklet detection)

It covers algorithm flow, equations, constants, and variable-level behavior.

---

## 1. End-to-End Signal Path

There are two pitch detection paths:

1. Preferred path (off main thread):
   - Microphone -> `AudioWorkletNode` (`pitch-processor.js`) -> MPM detection -> message to main thread

2. Fallback path (main thread):
   - Microphone -> `AnalyserNode` -> `getFloatTimeDomainData(...)` -> MPM detection in `app.js`

Both paths feed a common post-processing pipeline in `draw()`:

- octave-jump rejection
- Kalman smoothing
- silence handling
- swara mapping + hysteresis
- deviation in cents
- keyboard note highlight

---

## 2. Audio Capture and Frame Setup

### 2.1 Microphone setup (`startAudio()`)

`getUserMedia` is requested with:

- `echoCancellation: false`
- `autoGainControl: false`
- `noiseSuppression: false`

Why: pitch tracking requires raw-ish signal, not speech-enhanced signal.

### 2.2 Analysis frame size

- `analyser.fftSize = 2048`
- Main-thread fallback array: `dataArray = new Float32Array(analyser.fftSize)`

Worklet also uses:

- `this.buffer = new Float32Array(2048)`
- 50% overlap (`copyWithin(0, half)` where `half = 1024`)

At 44.1 kHz, 2048 samples is about 46.4 ms window; with 50% overlap the detector updates about every 23.2 ms.

---

## 3. Core Pitch Algorithm: McLeod Pitch Method (MPM)

Implemented twice with equivalent logic:

- `detectPitchMPM(buf, sampleRate)` in `app.js`
- `detectPitchMPM(buf)` in `pitch-processor.js` (uses global `sampleRate` in worklet)

### 3.1 Step A: RMS gate (silence/unvoiced rejection)

Compute:

\[
\text{rms} = \sqrt{\frac{1}{N} \sum_{i=0}^{N-1} x_i^2}
\]

If `rms < 0.008`, return no pitch:

- `{ pitch: -1, confidence: 0 }`

Constants:

- `MPM_RMS_THRESHOLD = 0.008` (main thread)
- `this.RMS_THRESHOLD = 0.008` (worklet)

### 3.2 Step B: NSDF computation

For lag \(\tau\), compute normalized squared difference:

\[
\text{NSDF}(\tau) = \frac{2 r(\tau)}{m(\tau)}
\]

Where:

\[
r(\tau) = \sum_j x_j x_{j+\tau}
\]
\[
m(\tau) = \sum_j (x_j^2 + x_{j+\tau}^2)
\]

Implementation details:

- `maxLag = SIZE / 2` (or `SIZE >> 1` in worklet)
- `nsdf` array length = `maxLag`
- loop limits use `maxLag - tau`

### 3.3 Step C: Skip initial positive NSDF region

The algorithm ignores the initial lobe near lag 0:

- start at index `1`
- while `nsdf[i] > 0`, advance

This defines `firstZeroCrossing` / `firstZero`.

### 3.4 Step D: Peak picking after first zero crossing

Collect peaks where:

- `nsdf[i] > nsdf[i-1]`
- `nsdf[i] >= nsdf[i+1]`
- `nsdf[i] > 0`

If no peaks: return no pitch.

### 3.5 Step E: Key-threshold peak selection

Find highest peak value \(p_{max}\), threshold:

\[
\text{threshold} = 0.93 \cdot p_{max}
\]

Then pick the **first** peak above threshold.

Reason: selecting first strong periodicity avoids octave errors from later harmonic peaks.

Constants:

- `MPM_KEY_THRESHOLD = 0.93` / `this.KEY_THRESHOLD = 0.93`

Fallback behavior if nothing passes threshold:

- choose last peak (`selectedPeak = peaks[peaks.length - 1]`)

### 3.6 Step F: Parabolic interpolation

Refine lag around selected peak index `T0` using 3-point parabola:

- `x1 = nsdf[T0 - 1]`
- `x2 = nsdf[T0]`
- `x3 = nsdf[T0 + 1]`
- `a = (x1 + x3 - 2*x2) / 2`
- `b = (x3 - x1) / 2`
- if `a != 0`: `T0 = T0 - b / (2*a)`

Then:

\[
\text{pitch} = \frac{f_s}{T_0}
\]

### 3.7 Step G: sanity range gate

Reject fundamentals outside:

- `< 50 Hz`
- `> 2000 Hz`

Return no pitch if outside bounds.

### 3.8 Step H: median-of-3 de-spike filter

Raw detected pitches are buffered:

- `lastRawPitches` (length capped at 3)

If 3 entries exist:

- sort
- return middle value

This removes single-frame outliers.

---

## 4. Worklet Transport and Freshness Handling

When worklet is active (`useWorklet = true`):

- worklet posts `{ pitch, confidence, timestamp: currentTime }`
- main thread stores latest in `workletDetection`

In `draw()`, main thread only consumes a result once:

- compare `workletDetection.timestamp` with `lastProcessedTimestamp`
- process only if different (`isNewDetection = true`)

This prevents re-processing the same detector output across multiple animation frames.

---

## 5. Post-Detection Smoothing and Stability Logic

After each accepted detection frame (`detection.pitch !== -1`), app applies multiple stability layers.

### 5.1 Kalman filter model (`PitchKalmanFilter`)

State vector:

- `x[0]`: pitch in Hz
- `x[1]`: pitch velocity in Hz/frame

Covariance:

- `P` is 2x2 matrix

Base parameters:

- `Q_base` (process noise): default `2.0`
- `R_base` (measurement noise): default `8.0`

#### Predict step

Constant-velocity model (`dt = 1`):

- `x[0] += x[1]`
- `x[1]` unchanged

Covariance update is hand-expanded in code.

#### Update step

Measurement = detected pitch.

Measurement noise depends on confidence:

\[
R = \frac{R_{base}}{\max(\text{confidence}, 0.1)}
\]

Higher confidence => lower measurement noise => stronger correction.

Kalman gain and state/covariance updates are scalarized for performance.

### 5.2 User smoothing slider -> process noise mapping

`sliderVal = parseFloat(smoothingInput.value)` where higher value means smoother UI.

Mapped as:

- `kalmanFilter.Q_base = 0.5 + (1 - sliderVal) * 8`

So:

- slider high -> lower process noise -> smoother/slower
- slider low -> higher process noise -> more responsive

### 5.3 Octave jump rejection

Before Kalman update, code checks:

\[
|\log_2(\frac{rawPitch}{kalmanPitch})| > 0.6
\]

`0.6` octaves is a very large jump threshold.

Behavior:

- increment `octaveErrorCount`
- if count > 4 consecutive frames: trust leap and reset Kalman to `rawPitch`
- else ignore spike and keep current Kalman pitch

This suppresses short-lived octave mis-tracks.

### 5.4 Gamaka-adaptive process noise

If jump is not octave-error class, code adapts prediction noise:

- `pitchDelta = abs(rawPitch - kalmanFilter.pitch)`
- `Q_scale = 3.0` if delta > 8 Hz, else `1.0`

Then:

- `predict(Q_scale)`
- `update(rawPitch, confidence)`

This makes fast melodic movements (gamakas) track quicker.

### 5.5 Silence timeout and reset

If detector returns no pitch:

- `silenceTimer += 16` (approx frame ms)
- if `silenceTimer > 300`:
  - `smoothedPitch = null`
  - clear `lastRawPitches`
  - `kalmanFilter.initialized = false`

Meaning: after about 300 ms unvoiced, tracking state resets.

---

## 6. Pitch-to-Swara Mapping (Just Intonation + Hysteresis)

Function: `updateClosestSwara(pitch, saFreq)`.

### 6.1 Reference system

`SWARAS` defines base swaras and just-intonation ratios:

- S = 1/1
- r = 16/15
- R = 9/8
- g = 6/5
- G = 5/4
- m = 4/3
- M = 45/32
- P = 3/2
- d = 8/5
- D = 5/3
- n = 9/5
- N = 15/8
- S' = 2/1 (excluded from nearest detection set)

Detection set excludes `S'` to avoid duplication.

### 6.2 Closest swara search

Compute current log ratio:

\[
L = \log_2\left(\frac{pitch}{saFreq}\right)
\]

Search candidate swaras across octaves `[-1, 0, 1]`.

For each candidate:

- `ratioLog = log2(swara.ratio * 2^octave)`
- minimize `abs(L - ratioLog)`

Store best:

- `closestSwaraObj`
- `closestOctave`
- `closestRatioLog`
- `minDiff`

### 6.3 Spatial hysteresis (anti-flicker)

Hysteresis constant:

- `HYSTERESIS_LOG2 = 25 / 1200` (25 cents)

If currently locked swara exists:

- `diffToCurrent = abs(L - currentSwaraRatioLog)`
- `diffToClosest = minDiff`

Switch only if:

- `diffToCurrent > diffToClosest + HYSTERESIS_LOG2`

This avoids rapid toggling near boundaries while preserving quick phrase tracking.

### 6.4 Deviation display

Deviation vs currently locked swara:

\[
\Delta_{cents} = (L - currentSwaraRatioLog) \cdot 1200
\]

Displayed as rounded cents with sign.

Color coding:

- <= 10 cents: green
- <= 25 cents: yellow
- else: red

---

## 7. Keyboard Note Mapping Path

Function: `updateKeyboard(pitch, saFreq)`.

### 7.1 Frequency -> MIDI conversion

`freqToMidi(freq)`:

\[
\text{midi} = 69 + 12\log_2\left(\frac{f}{440}\right)
\]

Then round to nearest integer for key highlight.

### 7.2 Keyboard hysteresis

If a key is already highlighted, switch only if:

- `abs(midiFloat - lastHighlightedMidi) >= 0.4`

This suppresses visual jitter between neighboring keys.

### 7.3 MIDI -> swara mapping relative to Sa

- `saMidi = round(freqToMidi(saFreq))`
- semitone distance mod 12 -> base swara via `SWARA_BASE_IDS`

Map:

- 0:S, 1:r, 2:R, 3:g, 4:G, 5:m, 6:M, 7:P, 8:d, 9:D, 10:n, 11:N

---

## 8. Timing and State Variables That Matter

### 8.1 Detector state

- `useWorklet`: choose worklet vs analyser fallback
- `pitchWorkletNode`: worklet node instance
- `workletDetection`: latest posted `{ pitch, confidence, timestamp }`
- `lastProcessedTimestamp`: dedupe worklet frames
- `dataArray`: analyser fallback sample buffer
- `lastRawPitches`: median-of-3 buffer (exists in both implementations)

### 8.2 Tracking state

- `smoothedPitch`: final pitch used by UI and history
- `silenceTimer`: ms-like counter for unvoiced duration
- `octaveErrorCount`: consecutive large-jump counter
- `kalmanFilter`: smoother and predictor

### 8.3 Swara lock state

- `currentSwaraKey`: current locked note id (example `R_0`)
- `currentSwaraRatioLog`: locked note reference in log2 space
- `currentSwaraObj`: locked swara object from `SWARAS`
- `currentOctave`: locked octave
- `HYSTERESIS_LOG2`: required closeness margin to switch

### 8.4 Reference tuning state

- `saFreqInput.value`: tonic (Sa) in Hz
- all swara detection is relative to this value

---

## 9. Why this stack is robust

The implementation combines complementary protections:

- RMS gating removes silent/unvoiced garbage
- NSDF + first-strong-peak rule reduces octave mistakes
- parabolic interpolation improves sub-sample lag precision
- median-of-3 removes one-frame outliers
- Kalman filtering balances smoothness and responsiveness
- octave-jump guard blocks implausible leaps
- gamaka-adaptive Q scaling preserves fast ornament tracking
- swara hysteresis prevents boundary flicker

---

## 10. Practical tuning knobs (current values)

- Detector sensitivity:
  - `RMS_THRESHOLD = 0.008`
  - `KEY_THRESHOLD = 0.93`
- Frequency validity range:
  - `50..2000 Hz`
- Octave jump detection:
  - `|log2 ratio| > 0.6`, accept only after `> 4` consecutive frames
- Silence reset:
  - `> 300 ms`
- Swara switch hysteresis:
  - `25 cents`
- Keyboard hysteresis:
  - `0.4 semitones`

These values are directly hard-coded in current source.

---

## 11. Exact per-frame flow in live mode

Inside `draw()` when not playing back:

1. Get new detection from worklet (or fallback analyser+MPM).
2. If voiced:
   - map smoothing slider to Kalman `Q_base`
   - init or update Kalman
   - run octave-jump rejection
   - run gamaka-adaptive predict/update
   - set `smoothedPitch`
3. If unvoiced:
   - advance `silenceTimer`
   - reset state after timeout
4. Push `{ pitch: smoothedPitch, time }` to `pitchHistory`.
5. Update UI from `smoothedPitch`:
   - Hz label
   - swara + octave + cents deviation
   - keyboard highlight
   - graph auto-follow

This is the exact operational note detection pipeline currently used by the app.
