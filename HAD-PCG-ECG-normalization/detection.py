import scipy.io
import numpy as np

file_path = r"Your_file_path_goes_here"  # Replace with your file path
mat_data = scipy.io.loadmat(file_path)
measure = mat_data['measure']

# loop through each cell in the measure array
for row in range(measure.shape[0]):
    for col in range(measure.shape[1]):
        entry = measure[row, col]
        if isinstance(entry, np.ndarray) and entry.size > 0:
            record = entry[0, 0]  #Unwrap the actual MATLAB struct

            label_field = record['label']
            try:
                label = label_field.item()
            except Exception:
                label = str(label_field)
            print(f"Label: {label}")

            if label == 'ECG':
                ecg_data = record['data']
                print("ECG data found!")
                print(ecg_data)

            if label == 'PCG':
                pcg_data = record['data']
                print("PCG data found!")
                print(pcg_data)
