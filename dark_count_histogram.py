import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

data = np.loadtxt("Dark_Count.csv", delimiter=",")

# Compute the baseline
baseline = np.average(data[:, :15], axis=1)

# Compute the differences between baseline and data
data_diff = (np.expand_dims(baseline, axis=1) - data).flatten()

# Create the histogram and get counts and bin edges
counts, bins, _ = plt.hist(data_diff, bins=np.arange(-15, 15), edgecolor='black', alpha=0.6, density=False)

# Fit a normal distribution to the data differences
mu, std = norm.fit(data_diff)

# Compute bin centers
bin_centers = (bins[:-1] + bins[1:]) / 2

# Compute the normal PDF over the bin centers
p = norm.pdf(bin_centers, mu, std)
    
# Scale the PDF to match the histogram
bin_width = bins[1] - bins[0]
p_scaled = p * counts.sum() * bin_width

# Plot the Gaussian fit over the histogram
plt.plot(bin_centers, p_scaled, 'r--', linewidth=2, label=f'Gaussian Fit\n$\mu={mu:.2f}$, $\sigma={std:.2f}$')

# Add labels and title
plt.title('Histogram with Gaussian Fit of Dark Count Differences')
plt.xlabel('Deviation from Baseline (ADC Units)')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.show()