import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad

# Load the data
lyso = "LYSO.csv"
bgo = "Datasets/DataR_run_green_02.csv"
csti = "Datasets/DataR_white_02.csv"
csti_1 = 13000
csti_2 = 17000
bgo_1 = 3500
bgo_2 = 4200
lyso_1 = 8000
lyso_2 = 9500
lyso_g = 20
bgo_g = 32
csti_g = 20

# Define a Gaussian function
def gaussian(x, amp, mean, stddev):
    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

# Load the data
data = np.loadtxt(bgo, delimiter=",")
data = data * (10 ** (-csti_g / 20))

# Calculate the baseline
baseline = np.average(data[:, :15], axis=1)

# Calculate the histogram
hist_values, bin_edges = np.histogram(np.sum(np.expand_dims(baseline, axis=1) - data, axis=1), bins=100)

# Get the bin centers
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Plot the full histogram to visualize the data
plt.figure(figsize=(10, 6))
plt.hist(np.sum(np.expand_dims(baseline, axis=1) - data, axis=1), bins=100, edgecolor='black', alpha=0.6, label="Full Histogram")
plt.title('Full Histogram of BGO Data')
plt.xlabel('ADC Units per Channel')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# Define the range of ADC units for fitting (adjust as needed)
x_min = csti_1  # Adjust based on your peak
x_max = csti_2

# Select the data within the specified range
mask = (bin_centers >= x_min) & (bin_centers <= x_max)
x_data = bin_centers[mask]
y_data = hist_values[mask]

# Debugging output to check the data
print(f"x_data: {x_data}")
print(f"y_data: {y_data}")
print(f"Initial guess: {np.max(y_data), np.mean(x_data), np.std(x_data)}")

# Check for NaNs and remove them
mask_nonan = ~np.isnan(x_data) & ~np.isnan(y_data)
x_data = x_data[mask_nonan]
y_data = y_data[mask_nonan]

# Ensure there is enough data for curve fitting
if len(x_data) > 3:
    # Fit the Gaussian curve to the specified section
    popt, _ = curve_fit(gaussian, x_data, y_data, p0=[np.max(y_data), np.mean(x_data), np.std(x_data)])

    # Extract the fitted parameters
    amp, mean, stddev = popt

    # Print the Gaussian equation with the fitted parameters
    print(f"Fitted Gaussian equation: f(x) = {amp:.2f} * exp(-((x - {mean:.2f})^2) / (2 * {stddev:.2f}^2))")
else:
    print("Not enough data points for curve fitting.")

# Fit the Gaussian curve to the specified section
popt, _ = curve_fit(gaussian, x_data, y_data, p0=[np.max(y_data), np.mean(x_data), np.std(x_data)])

# Extract the fitted parameters
amp, mean, stddev = popt

# Print the Gaussian equation with the fitted parameters
print(f"Fitted Gaussian equation: f(x) = {amp:.2f} * exp(-((x - {mean:.2f})^2) / (2 * {stddev:.2f}^2))")

# Define a function to integrate (the Gaussian function with the fitted parameters)
def fitted_gaussian(x):
    return gaussian(x, amp, mean, stddev)

# Compute the integral of the Gaussian over the range [x_min, x_max]
integral, _ = quad(fitted_gaussian, mean - 2 * stddev, mean + 2 * stddev)

# Calculate the average value (integral divided by the width of the range)
average_value = integral / (4*stddev)

# Print the integral and the average value
print(f"Integral under the Gaussian curve: {integral:.2f}")
print(f"Average value of the Gaussian curve over [{x_min}, {x_max}]: {average_value:.2f}")

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.hist(np.sum(np.expand_dims(baseline, axis=1) - data, axis=1), bins=100, edgecolor='black', alpha=0.6, label="Histogram")

# Plot the fitted Gaussian curve over the specified section
x_fit = np.linspace(x_min, x_max, 1000)
y_fit = gaussian(x_fit, *popt)
plt.plot(x_fit, y_fit, color='red', lw=2, label=f'Gaussian fit\nAmp={amp:.2f}, Mean={mean:.2f}, Stddev={stddev:.2f}')

# Add titles and labels
plt.title('Integrated Waveform Histogram with Gaussian Fit (Peak Section)')
plt.xlabel('ADC Units per Channel')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.show()

# Define the energy of Cs-137 gamma rays
cs137_energy_keV = 662  # keV

# Calculate the light yield in ADC units per keV
light_yield_per_keV = mean / cs137_energy_keV

# Print the light yield
print(f"Light yield: {light_yield_per_keV:.2f} ADC units per keV")