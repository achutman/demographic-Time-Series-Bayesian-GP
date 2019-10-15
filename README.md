# demographicTimeSeriesBayesianGPy
Use Gaussian Processes (GPs) based hypothesis test for time-series data

scripts/timeSeriesModelGPyExample.py is an example script to train independent GPs to model two time-series data representing demographic data. The script uses modules and class defined in scripts/timeSeriesModelGPyUtils.py, scripts/timeSeriesModelGPy.py, and scripts/timeSeriesModelGPyDiff.py.

data/intCountsIoOiNormPerYr.csv is an example dataset, which is used by scripts/timeSeriesModelGPyExample.py.

Example outputs are shown in outputs/...

The example is tested using Python 3.7.3, sklearn 0.21.2, and GPy.
For GPy installation, please refer to https://github.com/SheffieldML/GPy
For details on GP based hypothesis test, please refer to http://proceedings.mlr.press/v38/benavoli15.pdf
