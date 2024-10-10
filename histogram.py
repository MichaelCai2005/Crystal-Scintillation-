import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file (replace 'your_file.csv' with the actual file path)
# df = pd.read_csv('DataR_run.csv')
# data = df.iloc[0:76353, 0:248]
data = np.loadtxt("Datasets/DataR_white_02.csv", delimiter=",")

# data_values = data.values.flatten()
baseline = np.average(data[:, :15], axis=1)
plt.figure(figsize=(10,6))
# plt.plot(data[0, :])
# baseline = np.average(data[:, 240:])
plt.hist(np.sum(np.expand_dims(baseline, axis=1) - data, axis=1), bins=100, edgecolor='black')
# plt.hist(data_values, bins=50, edgecolor='black')
plt.title('Integrated Waveform Histogram of "CsTi Sample"')
plt.xlabel('ADC Units per Channel')
plt.ylabel('Count')
plt.grid(True)
plt.show()