# Data Analysis Project

## Overview

This project contains a collection of Python scripts designed for analyzing waveform and dark count data. The scripts perform tasks such as computing average waveforms, generating histograms, fitting Gaussian and exponential models, and visualizing single waveform samples. All the necessary code is included in this repository.

## Table of Contents

- [Overview](#overview)
- [Contents](#contents)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [average-waveform.py](#average-waveformpy)
  - [dark_count_histogram.py](#dark_count_histogrampy)
  - [fit-histogram.py](#fit-histogrampy)
  - [histogram.py](#histogrampy)
  - [singular-waveform.py](#singular-waveformpy)
- [Datasets](#datasets)
- [Contact](#contact)
- [License](#license)

## Contents

- `average-waveform.py`: Computes the average waveform from multiple data runs and fits an exponential model.
- `dark_count_histogram.py`: Generates a histogram of dark count differences and fits a Gaussian distribution.
- `fit-histogram.py`: Fits a Gaussian curve to a specified section of the histogram and calculates related statistics.
- `histogram.py`: Creates an integrated waveform histogram for the "CsTi Sample".
- `singular-waveform.py`: Plots a single waveform from the dataset.

## Prerequisites

Ensure you have the following installed on your system:

- Python 3.6 or higher
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [SciPy](https://www.scipy.org/)

You can install the required Python packages using `pip`:

```bash
pip install numpy pandas matplotlib scipy
```

## Installation
1. Clone the repository
2. Place data sets in appropriate directories

```
your-repo-name/
├── Datasets/
│   ├── DataR_run_black_02.csv
│   ├── DataR_run_green_02.csv
│   ├── DataR_white_02.csv
│   └── Dark_Count.csv
├── average-waveform.py
├── dark_count_histogram.py
├── fit-histogram.py
├── histogram.py
└── singular-waveform.py
```
## Datasets
The required datasets are not included in this repository. Please ensure that the following CSV files are stored locally in the `Datasets/` directory:

- `DataR_run_black_02.csv`
- `DataR_run_green_02.csv`
- `DataR_white_02.csv`
- `Dark_Count.csv`
Ensure that the file paths within the scripts correctly point to these datasets.

## Contact 
For any questions or issues, please contact:

Email: mhc2167@columbia.edu

## License
This project is licensed under the MIT License.

