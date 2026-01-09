import os
import numpy as np
import scipy.io
from scipy.signal import resample
from scipy.fft import fft, ifft, fftfreq

# Coimbra unhealthy dataset path and output path
# Example: HAD-Normalization\HAD-PCG-ECG-normalization\CycleGAN\data\coimbra_unhealthy
# # Example: HAD-Normalization\HAD-PCG-ECG-normalization\CycleGAN\data\COIMBRA_CVD_NORMALIZED
UNHEALTHY_INPUT_PATH = r"YourPATH_to_unhealthy_dataset"  # Replace with your Coimbra unhealthy dataset path
OUTPUT_UNHEALTHY_NORMALIZED = r"YourPATH_to_normalized_files"  # Replace with your output path for normalized files

def fft_bandpass(signal, fs, lowFreq, highFreq):
    # Figuring out how many poinits are in the signal
    num_points = len(signal)
    # Creating a frequence bins
    freqs = fftfreq(num_points, 1/fs)
    signal_fft = fft(signal)

    keep_freqs = []
    for f in freqs:
        if abs(f) >= lowFreq and abs(f) <= highFreq:
            keep_freqs.append(True)  
        else:
            keep_freqs.append(False)

    # applyting the mask to keep the unwanted frequencies to zero
    filtered_fft = signal_fft.copy()
    for i in range(num_points):
        if not keep_freqs[i]:
            filtered_fft[i] = 0
    filtered_signal = ifft(filtered_fft)

    return np.real(filtered_signal)

def normalize_and_filter(signal, fs, lowcut, highcut):
    signal = signal - np.mean(signal)
    std = np.std(signal)
    if std != 0:
        signal = signal / std
    return fft_bandpass(signal, fs, lowcut, highcut)

def resample_signal(signal, orig_fs, target_fs):
    target_len = int(len(signal) * target_fs / orig_fs)
    return resample(signal, target_len)

def process_file(file_path, output_path):
    mat = scipy.io.loadmat(file_path)

    if 'measure' not in mat:
        print(f"Skipping {os.path.basename(file_path)}: 'measure' key not found.")
        return

    measure = mat['measure']
    ecg, pcg, fs = None, None, None

    for i in range(measure.shape[0]):
        for j in range(measure.shape[1]):
            entry = measure[i, j]
            if isinstance(entry, np.ndarray) and entry.size > 0:
                record = entry[0, 0]
                try:
                    label = record['label'].item()
                    if label == 'ECG' and ecg is None:
                        ecg = record['data'].flatten()
                        fs = int(record['Rate'].item())
                    elif label == 'PCG' and pcg is None:
                        pcg = record['data'].flatten()
                        if fs is None:
                            fs = int(record['Rate'].item())
                except Exception as e:
                    continue

    if ecg is None or pcg is None or fs is None:
        print(f"Skipping {os.path.basename(file_path)}: Could not extract ECG/PCG.")
        return

    # Resample to 8000 Hz
    target_fs = 8000
    ecg_rs = resample_signal(ecg, fs, target_fs)
    pcg_rs = resample_signal(pcg, fs, target_fs)

    ecg_filtered = fft_bandpass(ecg_rs, target_fs, 0.5, 40)
    pcg_filtered = fft_bandpass(pcg_rs, target_fs, 25, 150)

    ecg_processed = (ecg_filtered - np.mean(ecg_filtered)) / (np.std(ecg_filtered) + 1e-10)
    pcg_processed = (pcg_filtered - np.mean(pcg_filtered)) / (np.std(pcg_filtered) + 1e-10)

    save_dict = {
        'ECG': ecg_processed,
        'PCG': pcg_processed,
        'fs': target_fs,
        'original_file': os.path.basename(file_path)
    }

    output_file = os.path.join(output_path, f"normalized_{os.path.basename(file_path)}")
    scipy.io.savemat(output_file, save_dict)
    print(f"Saved: {output_file}")

def main():
    os.makedirs(OUTPUT_UNHEALTHY_NORMALIZED, exist_ok=True)

    all_mat_files = []
    for root, dirs, files in os.walk(UNHEALTHY_INPUT_PATH):
        for file in files:
            if file.endswith(".mat"):
                all_mat_files.append(os.path.join(root, file))

    print(f"Found {len(all_mat_files)} .mat files in {UNHEALTHY_INPUT_PATH}")

    for fpath in all_mat_files:
        try:
            process_file(fpath, OUTPUT_UNHEALTHY_NORMALIZED)
        except Exception as e:
            print(f" Error processing {fpath}: {e}")

    print("All unhealthy Coimbra files processed and saved.")

if __name__ == "__main__":
    main()
