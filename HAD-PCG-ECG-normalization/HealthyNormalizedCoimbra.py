import os
import numpy as np
import scipy.io
from scipy.signal import resample
from scipy.fft import fft, ifft, fftfreq


# Coimbra healthy dataset path and output path
# Example: HAD-Normalization\HAD-PCG-ECG-normalization\CycleGAN\data\coimbra_healthy
# Example: HAD-Normalization\HAD-PCG-ECG-normalization\CycleGAN\data\coimbra_ephnogram_healthy
COIMBRA_HEALTHY_PATH = r"YourPATH" # Replace with your Coimbra healthy dataset path
OUTPUT_COMBINED_PATH = r"YourPATH" # Replace with your output path for normalized files


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


def normalize_and_filter(signal, fs, lowFreq, highFreq):
    signal = signal - np.mean(signal) # removes the DC offset 
    signal = fft_bandpass(signal, fs, lowFreq, highFreq)
    # center it again after filtering just in case 
    signal = signal - np.mean(signal)
    signal_std = np.std(signal)
    if signal_std != 0:
        signal = signal / signal_std  # Normalizing the filtered signal
    return signal

def change_frequency(signal, orig_fs, target_fs):
    new_length = int(len(signal) * target_fs / orig_fs) # figuring out how many samples we need in the new signal
    return resample(signal, new_length)

#process one .mat file
def process_one_file(file_path, output_path):
    data = scipy.io.loadmat(file_path)

    if 'measure' not in data:
        print(f"no 'measure' key in {os.path.basename(file_path)}")
        return

    ecg = None
    pcg = None
    fs = None

    # Looping through the data to find ECG and PCG
    measure = data['measure']
    for i in range(measure.shape[0]):
        for j in range(measure.shape[1]):
            entry = measure[i, j]
            if isinstance(entry, np.ndarray) and entry.size > 0:
                record = entry[0, 0]
                try:
                    label = record['label'].item()
                    if label == 'ECG' and ecg is None:
                        ecg = record['data'].flatten() # Getting the ECG data
                        fs = int(record['Rate'].item()) #Getting the sampling frequency
                    elif label == 'PCG' and pcg is None:
                        pcg = record['data'].flatten() # Getting the PCG data
                        if fs is None:
                            fs = int(record['Rate'].item())
                except Exception as e:
                    continue

    if ecg is None or pcg is None or fs is None:
        print(f"Skipping {os.path.basename(file_path)}: Could not extract ECG/PCG.")
        return

    # Resample to 8000 Hz
    target_fs = 8000
    ecg_rs = change_frequency(ecg, fs, target_fs)
    pcg_rs = change_frequency(pcg, fs, target_fs)

    # Cleaning and normalizing the signals 
    ecg_clean = normalize_and_filter(ecg_rs, target_fs, 0.5, 40)
    pcg_clean = normalize_and_filter(pcg_rs, target_fs, 25, 150)

    save_data = {
        'ECG': ecg_clean,
        'PCG': pcg_clean,
        'fs': target_fs,
        'original_file': os.path.basename(file_path)
    }

    # Creating the output filename
    output_file = os.path.join(output_path, f"normalized_{os.path.basename(file_path)}")
    scipy.io.savemat(output_file, save_data)
    print(f"Saved: {output_file}")

def main():
    # First, ensure that the output directory exists 
    if not os.path.exists(OUTPUT_COMBINED_PATH):
        os.makedirs(OUTPUT_COMBINED_PATH)

    # Recursively find all .mat files
    all_mat_files = []
    for root, dirs, files in os.walk(COIMBRA_HEALTHY_PATH):
        for file in files:
            if file.endswith(".mat"):
                all_mat_files.append(os.path.join(root, file))

    print(f"Found {len(all_mat_files)} .mat files in {COIMBRA_HEALTHY_PATH} (recursively)")

    #Processing each file
    for fpath in all_mat_files:
        try:
            process_one_file(fpath, OUTPUT_COMBINED_PATH)
        except Exception as e:
            print(f"Error with {fpath}: {e}")

    print("Finished processing all files.")

if __name__ == "__main__":
    main()