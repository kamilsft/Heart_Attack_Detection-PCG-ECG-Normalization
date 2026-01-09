import os
import scipy.io
import numpy as np

# This script checks if the ECG and PCG signals in the Coimbra dataset are normalized.
# It prints the mean and standard deviation of each signal, and checks if they are within the expected range.
# Example of the path: HAD-Normalization\HAD-PCG-ECG-normalization\CycleGAN\data\COIMBRA_CVD_NORMALIZED
FOLDER = r"YourPATH" # Replace with your folder path

for file in os.listdir(FOLDER):
    if not file.endswith(".mat"):
        continue
    try:
        data = scipy.io.loadmat(os.path.join(FOLDER, file))
        ecg = data['ECG'].flatten()
        pcg = data['PCG'].flatten()

        # read FS if available
        fs = data['fs'][0][0] if 'fs' in data else 'N/A'

        ecg_mean, ecg_std = np.mean(ecg), np.std(ecg)
        pcg_mean, pcg_std = np.mean(pcg), np.std(pcg)

        print(f"{file}")
        print(f"  FS: {fs} Hz")
        print(f"  ECG mean: {ecg_mean:.5f}, std: {ecg_std:.5f}")
        print(f"  PCG mean: {pcg_mean:.5f}, std: {pcg_std:.5f}")
        if not (0.9 < ecg_std < 1.1) or not (0.9 < pcg_std < 1.1):
            print("NOT NORMALAZED\n")

    except Exception as e:
        print(f"Error in {file}: {e}")
