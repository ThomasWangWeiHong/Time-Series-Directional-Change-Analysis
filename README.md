# Time-Series-Directional-Change-Analysis

This repository contains a simple Python implementation of the directional change analysis techniques which are proposed in [1], and inspired by another GitHub repository 'JurgenPalsma/dcgenerator' (https://github.com/JurgenPalsma/dcgenerator).

## Introduction

Directional Change (DC) analysis is a paradigm proposed by the authors in [1] for the analysis of financial time series. 

In the traditional time series analysis paradigm, one would sample prices at fixed intervals, whereas the DC paradigm is essentially a data-driven approach where the data informs the algorithm when to sample prices. By looking at price changes from another perspective, it is believed that one can extract new information from data that complements what is oberserved under the traditional time series analysis paradigm. This new information can be utilized by machine learning techniques in order to infer regime information about the market, which in turn helps in the development of algorithmic trading strategies.

## Sample Results

The following results are obtained using a truncated version of real historical EUR-USD exchange rate data for the period of September 2023 (first 1000 tick data points), which is provided by TrueFX (https://www.truefx.com/). In order to download the complete dataset, one must first register for an account (Free-of-Charge) with TrueFX before accessing the 'Historical Downloads' section (https://www.truefx.com/truefx-historical-downloads/) to download the dataset.

The original time series plot is shown in the figure below. The time series can additionally be smoothened using a kalman filter first before performing the DC analysis for better results.

![Original Time Series Plot](https://github.com/ThomasWangWeiHong/Time-Series-Directional-Change-Analysis/blob/main/assets/EURUSD-2023-09%20Time%20Series%20Plot.jpg)

The thresholds for event detection can then be set by the user (default threshold value is 0.0001), after which the DC analysis can be performed. The resulting annotated time series plot and its animated version are shown below.

![Annotated Time Series Plot](https://github.com/ThomasWangWeiHong/Time-Series-Directional-Change-Analysis/blob/main/assets/Annotated%20EURUSD-2023-09%20Plot.jpg)

![Annotated Time Series Animation](https://github.com/ThomasWangWeiHong/Time-Series-Directional-Change-Analysis/blob/main/assets/Annotated%20EURUSD-2023-09%20Animation.gif)

In addition, the Total Price Movement (TMV) variable and Time for Completion of a Trend (T) variable are then computed from the data, normalized, and plotted in order to visualize how these two variables can be used to identify different regimes in the market. The implementation in this repository uses the event class to colour the scatterplot points instead of using the regime information, so one may wish to modify the source code to use regime information instead when it is available. For more details on the computation of the TMV and T variables, one can consult [1].

The TMV-T feature space for the truncated data is illustrated below:

![TMV-T Feature Space](https://github.com/ThomasWangWeiHong/Time-Series-Directional-Change-Analysis/blob/main/assets/EURUSD-2023-09%20Indicator%20Feature%20Space%20Plot.jpg)

## Installation:

1. Ensure that Python 3 (tested on Python 3.8 and above) is installed in your system
2. Run the following command in order to install all the necessary packages: `pip install -r requirements.txt` (Make sure that you have the administrative rights to install the packages)
3. Enjoy!

## References

[1]  Chen, J., & Tsang, E. P. (2020). Detecting regime change in computational finance: data science, machine learning and algorithmic trading. CRC press.
