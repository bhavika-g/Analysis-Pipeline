## Analysis for Ephys+Calcium Imaging Data
This repository provides a pipeline to build a machine learning classifier for ephys+calcium imaging data using manual feature extraction. 

Input Format: 1D signals stored in .npz files

Goal: Predict or classify activity patterns using interpretable ML methods

## ⚙️  Signal Processing Methods Used:
1. Noise filtering
2. Average Subtraction
3. Peak Detection
4. Onset and End Detection
   
## 📊 Features Used:
1. Frequency domain Based (Fourier Transform, Wavelet Transform)
2. Time domain based (Ampltitude, Max, Peak, Mean, STD, Kurtosis, Skewness)
3. Higuchi Fractal Dimension Analysis
4. Detrended Fluctuation Analysis (DFA)

## 📈 Other Statistical Methods:
1. Principle Component Analysis
2. Singular Value Decomposition
3. Gaussian Mixture Models
4. SHAP values

## 🤖 Classifiers
1. Random Forest
2. XGBoost
