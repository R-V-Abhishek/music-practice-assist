import numpy as np
import librosa
import scipy.signal

def detect_tonic_stft(audio_path, sr=22050):
    y, sr = librosa.load(audio_path, sr=sr)

    # Compute STFT
    D = librosa.stft(y, n_fft=8192, hop_length=1024)
    magnitude = np.abs(D)

    # Average spectrum across time
    avg_spectrum = np.mean(magnitude, axis=1)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=8192)

    # Limit to vocal range
    valid = np.where((freqs > 80) & (freqs < 1000))[0]
    freqs = freqs[valid]
    spectrum = avg_spectrum[valid]

    # Find spectral peaks
    peaks, _ = scipy.signal.find_peaks(spectrum, height=np.max(spectrum)*0.1)
    peak_freqs = freqs[peaks]

    if len(peak_freqs) == 0:
        return None

    # Harmonic summation scoring
    scores = []
    for f in peak_freqs:
        harmonic_sum = 0
        for h in range(1, 6):  # first 5 harmonics
            target = f * h
            idx = np.argmin(np.abs(freqs - target))
            harmonic_sum += spectrum[idx]
        scores.append(harmonic_sum)

    scores = np.array(scores)

    # Best harmonic candidate
    tonic_candidate = peak_freqs[np.argmax(scores)]

    # Fold into practical octave (Carnatic vocal range)
    while tonic_candidate < 100:
        tonic_candidate *= 2
    while tonic_candidate > 400:
        tonic_candidate /= 2

    tonic_note = librosa.hz_to_note(tonic_candidate)

    return tonic_candidate, tonic_note


if __name__ == "__main__":
    tonic = detect_tonic_stft("GSharp2.mp3")
    print("Detected STFT-based Tonic:", tonic)