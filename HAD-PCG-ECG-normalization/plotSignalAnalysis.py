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

def plot_ecg_pcg(ecg, pcg, fs):
    # Normalize and filter
    ecg = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-10)
    pcg = (pcg - np.mean(pcg)) / (np.std(pcg) + 1e-10)

    ecg_filtered = fft_bandpass(ecg, fs, 0.5, 40)
    pcg_filtered = fft_bandpass(pcg, fs, 25, 150)

    print("---- Normalization Check ----")
    print(f"ECG mean: {np.mean(ecg):.6f}")
    print(f"ECG std:  {np.std(ecg):.6f}")
    print(f"PCG mean: {np.mean(pcg):.6f}")
    print(f"PCG std:  {np.std(pcg):.6f}")
    print("------------------------------")

    # Prepare segments (2s and 10s)
    ecg_samples_2s = min(int(2 * fs), len(ecg_filtered))
    ecg_samples_10s = min(int(10 * fs), len(ecg_filtered))
    pcg_samples_2s = min(int(2 * fs), len(pcg_filtered))
    pcg_samples_10s = min(int(10 * fs), len(pcg_filtered))
    
    ecg_time_2s = np.linspace(0, 2, ecg_samples_2s)
    ecg_time_10s = np.linspace(0, 10, ecg_samples_10s)
    pcg_time_2s = np.linspace(0, 2, pcg_samples_2s)
    pcg_time_10s = np.linspace(0, 10, pcg_samples_10s)

    # Plot
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.plot(ecg_time_2s, ecg_filtered[:ecg_samples_2s], color='purple')
    plt.title("ECG (0.5–40 Hz) - 2s Detail")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(ecg_time_10s, ecg_filtered[:ecg_samples_10s], color='purple')
    plt.title("ECG (0.5–40 Hz) - 10s Overview")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(pcg_time_2s, pcg_filtered[:pcg_samples_2s], color='blue')
    plt.title("PCG (25–150 Hz) - 2s Detail")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(pcg_time_10s, pcg_filtered[:pcg_samples_10s], color='blue')
    plt.title("PCG (25–150 Hz) - 10s Overview")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# put a directory path the one you want to run the plot analysis on
# Example: HAD-Normalization\HAD-PCG-ECG-normalization\CycleGAN\data\COIMBRA_CVD_NORMALIZED\normalized_Measure_20101119_C62S0_HDF.mat
def main():
    file_path = r"YourPATH"  # Replace with your file path
    
    mat = scipy.io.loadmat(file_path)

    ecg = mat['ECG'].flatten()
    pcg = mat['PCG'].flatten()
    fs = int(mat['fs'][0][0])  # 8000

    plot_ecg_pcg(ecg, pcg, fs)

if __name__ == '__main__':
    main()