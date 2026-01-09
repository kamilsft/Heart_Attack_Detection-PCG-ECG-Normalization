import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

def fft_bandpass(signal, fs, lowcut, highcut):
    # Applies a bandpass filter using FFT
    n = len(signal)
    freqs = fftfreq(n, 1/fs)
    spectrum = fft(signal)
    mask = (np.abs(freqs) >= lowcut) & (np.abs(freqs) <= highcut)
    return np.real(ifft(spectrum * mask))

def normalize_signal(signal):
    # Normalizes signal 

    # Remove mean (DC offset)
    signal = signal - np.mean(signal)
    
    # Scale to -0.75 to 0.75 range
    max_abs_val = np.max(np.abs(signal))
    if max_abs_val == 0: # Avoid division by zero if signal is flat
        return signal
    return 0.75 * signal / max_abs_val

def process_and_normalize_ephnogram_signal(signal, fs, signal_type):
    normalized_signal = normalize_signal(signal)

    # Apply FFT bandpass filtering
    if signal_type == 'ECG':
        filtered_signal = fft_bandpass(normalized_signal, fs, 0.5, 40)
    elif signal_type == 'PCG':
        filtered_signal = fft_bandpass(normalized_signal, fs, 25, 150)
    else:
        filtered_signal = normalized_signal

    # Normalize again after filtering
    filtered_signal -= np.mean(filtered_signal)
    std = np.std(filtered_signal)
    if std != 0:
        filtered_signal /= std

    return filtered_signal


def plot_ecg_pcg(ecg_filtered, pcg_filtered, fs, file_name=""):
    """
    Plots the filtered ECG and PCG signals (2s and 10s views).
    Takes already filtered signals as input.
    """
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
    plt.suptitle(f"Processed Ephnogram: {file_name}", fontsize=16)

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

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()


def main():
    # --- Configuration ---
    # Path to the directory containing Ephnogram .mat files
    #Example for Ephnogram input path: HAD-Normalization\HAD-PCG-ECG-normalization\ephnogram-a-simultaneous-electrocardiogram-and-phonocardiogram-database-1.0.0\MAT
    EPHNOGRAM_INPUT_PATH = r"YOUR_EPHNOGRAM_INPUT_PATH_HERE"  # Replace with your Ephnogram input path
    
    # Output directory for normalized Ephnogram files
    # HAD-PCG-ECG-normalization\CycleGAN\data\coimbra_ephnogram_healthy
    OUTPUT_DIR = r"YOUR_OUTPUT_DIR_HERE"  # Replace with your output directory path

    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory created/exists: {OUTPUT_DIR}")

    # List all .mat files in the input directory
    mat_files = [f for f in os.listdir(EPHNOGRAM_INPUT_PATH) if f.endswith(".mat")]

    if not mat_files:
        print(f"No .mat files found in {EPHNOGRAM_INPUT_PATH}. Exiting.")
        return

    for file_name in sorted(mat_files):
        input_file_path = os.path.join(EPHNOGRAM_INPUT_PATH, file_name)
        print(f"\nProcessing file: {file_name}")

        try:
            mat = scipy.io.loadmat(input_file_path)

            # Data Extraction (Ephnogram specific) 
            # Based on your provided 'main' function for Ephnogram,
            # it assumes direct keys 'ECG', 'PCG', and 'fs'.
            if 'ECG' not in mat or 'PCG' not in mat or 'fs' not in mat:
                print(f"  Skipping {file_name}: Missing 'ECG', 'PCG', or 'fs' keys.")
                continue

            ecg_raw = mat['ECG'].flatten()
            pcg_raw = mat['PCG'].flatten()
            fs = int(mat['fs'][0][0])  # Assuming fs is stored as a 1x1 array

            print(f" Loaded: FS = {fs}Hz, ECG length = {len(ecg_raw)}, PCG length = {len(pcg_raw)}")

            # Ensure ECG and PCG have the same length
            if len(ecg_raw) != len(pcg_raw):
                min_len = min(len(ecg_raw), len(pcg_raw))
                ecg_raw = ecg_raw[:min_len]
                pcg_raw = pcg_raw[:min_len]
                print(f" Adjusted signal lengths to {min_len} due to mismatch.")

            # Process (Filter and Normalize) Signals 
            # Your original plot_ecg_pcg function normalized before filtering.
            # We'll follow that structure for consistency 
            ecg_processed = process_and_normalize_ephnogram_signal(ecg_raw, fs, 'ECG')
            pcg_processed = process_and_normalize_ephnogram_signal(pcg_raw, fs, 'PCG')
            
            print(f" Signals processed and normalized. FS remains {fs}Hz.")
            # Save Processed Data to New .mat File 
            output_file_name = f"normalized_{file_name}"
            output_file_path = os.path.join(OUTPUT_DIR, output_file_name)

            # Prepare data for saving
            # We'll save ECG, PCG, and FS into the new .mat file.
            # Using original keys for clarity.
            save_data = {
                'ECG': ecg_processed,
                'PCG': pcg_processed,
                'fs': fs,
                'original_file': file_name 
            }
            scipy.io.savemat(output_file_path, save_data, do_compression=True)
            print(f"  Saved processed data to: {output_file_path}")

        except Exception as e:
            print(f" Error processing {file_name}: {e}")

    print("\nEphnogram normalization and saving complete.")

if __name__ == '__main__':
    main()