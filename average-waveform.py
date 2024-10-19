import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from scipy.optimize import curve_fit
#C:\Users\micha\OneDrive\2 School\Crystal Scintillation\Crystal-Scintillation-\Images
data = np.loadtxt("Datasets/DataR_run_white.csv", delimiter=",")
# print(data)
# first_row = data[0]
# add = data[0] + data[1]
# print(add)

rows = data.shape[0]
print(rows)
total_waveform = np.zeros(len(data[0]))

for i in range(rows):
    total_waveform += -data[i] + 3612

average_waveform = total_waveform/rows

x_values = np.arange(len(average_waveform)) * 4

# Define the exponential model y = A * exp(-x / tau)
def exp_model(x, A, tau):
    return A * np.exp(-x / tau)

# Select the data range from x = 162 to x = 620
x_fit_range = np.logical_and(x_values >= 1000, x_values <= 1750)
x_fit = x_values[x_fit_range]
y_fit = average_waveform[x_fit_range]

# Perform curve fitting
popt, pcov = curve_fit(exp_model, x_fit, y_fit, p0=(2200, 100))  # Initial guess for A and tau

# Extract fitted parameters
A_fitted, tau_fitted = popt
print(f"Fitted parameters: A = {A_fitted}, tau = {tau_fitted}")

# Generate the fitted curve
y_fitted = exp_model(x_fit, A_fitted, tau_fitted)

# Plot the original data
plt.plot(x_values, average_waveform, marker='o', label='Average Waveform')

# Plot the fitted exponential curve
plt.plot(x_fit, y_fitted, color='red', label=f'Fitted exponential curve: $e^{{-x/{tau_fitted:.2f}}}$')

# Add labels, title, legend, and grid
plt.title('Average Waveform with Exponential Fit')
plt.xlabel('Time (NanoSeconds)')
plt.ylabel('ADC Channels')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
