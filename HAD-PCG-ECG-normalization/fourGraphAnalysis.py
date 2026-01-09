import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

def fft_bandpass(signal, fs, lowcut, highcut):
    n = len(signal)
    freqs = fftfreq(n, 1/fs)
    spectrum = fft(signal)
    mask = (np.abs(freqs) >= lowcut) & (np.abs(freqs) <= highcut)
    return np.real(ifft(spectrum * mask))

def plot_ecg_pcg(ecg, pcg, fs, title_prefix):
    ecg = ecg - np.mean(ecg)
    ecg = 0.75 * ecg / (np.max(np.abs(ecg)) + 1e-10)
    ecg_filtered = fft_bandpass(ecg, fs, 0.5, 40)

    pcg = pcg - np.mean(pcg)
    pcg = 0.75 * pcg / (np.max(np.abs(pcg)) + 1e-10)
    pcg_filtered = fft_bandpass(pcg, fs, 25, 150)

    #Updated to use 2s and 10s intervals
    ecg_samples_2s = min(int(2 * fs), len(ecg_filtered))
    ecg_samples_10s = min(int(10 * fs), len(ecg_filtered))
    pcg_samples_2s = min(int(2 * fs), len(pcg_filtered))
    pcg_samples_10s = min(int(10 * fs), len(pcg_filtered))

    ecg_time_2s = np.linspace(0, 2, ecg_samples_2s)
    ecg_time_10s = np.linspace(0, 10, ecg_samples_10s)
    pcg_time_2s = np.linspace(0, 2, pcg_samples_2s)
    pcg_time_10s = np.linspace(0, 10, pcg_samples_10s)

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.plot(ecg_time_2s, ecg_filtered[:ecg_samples_2s], color='purple')
    plt.title(f"{title_prefix} ECG (2s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(ecg_time_10s, ecg_filtered[:ecg_samples_10s], color='purple')
    plt.title(f"{title_prefix} ECG (10s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(pcg_time_2s, pcg_filtered[:pcg_samples_2s], color='blue')
    plt.title(f"{title_prefix} PCG (2s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(pcg_time_10s, pcg_filtered[:pcg_samples_10s], color='blue')
    plt.title(f"{title_prefix} PCG (10s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    folder_path = r"Path_goes_here"  # Change as needed

    mat_files = [f for f in os.listdir(folder_path) if f.endswith(".mat")]
    if not mat_files:
        print("No .mat files found.")
        return

    for file in sorted(mat_files):
        try:
            print("Processing: {file}")
            path = os.path.join(folder_path, file)
            mat = scipy.io.loadmat(path)

            if 'measure' not in mat:
                print("Skipped: no 'measure' key")
                continue

            measure = mat['measure']

            ecg_entry = measure[0, 2][0, 0]
            pcg_entry = measure[1, 2][0, 0]

            ecg = ecg_entry['data'].flatten()
            pcg = pcg_entry['data'].flatten()
            fs_ecg = int(ecg_entry['Rate'].item())
            fs_pcg = int(pcg_entry['Rate'].item())

            if fs_ecg != 44100 or fs_pcg != 44100 or len(ecg) != len(pcg):
                print("Mismatched or non-44100Hz signals. Skipping.")
                continue

            plot_ecg_pcg(ecg, pcg, fs_ecg, title_prefix=file)

        except Exception as e:
            print("Error with {file}: {e}")

if __name__ == '__main__':
    main()