# Credit Card Fraud Detection Using Autoencoders

This project implements an anomaly detection system for credit card fraud using **Autoencoders**. By training the model exclusively on non-fraudulent transactions, it learns to reconstruct "normal" patterns. When a fraudulent transaction occurs, the high reconstruction error allows the system to flag it as an anomaly.

## Methodology

### 1. Data Preprocessing
* **Feature Engineering**: The `Time` and `Amount` features are log-scaled to reduce the impact of outliers.
* **Normalization**: All features are scaled to a range of (0, 1) using `MinMaxScaler`.
* **Data Splitting**: The training and validation sets consist entirely of non-fraudulent transactions. The test set is a combination of remaining non-fraud and all fraud transactions.

### 2. Model Architecture
Two different architectures were compared to evaluate performance:

* **First Autoencoder**: A shallow model consisting of an input layer, an encoding layer with 14 units, and a decoding layer.
* **Second Autoencoder**: A deeper model featuring two encoding layers (14 and 10 units) and two decoding layers.


[Image of Autoencoder architecture]


### 3. Fraud Detection Mechanism
The model classifies transactions based on the **Reconstruction Error** (Mean Squared Error). A threshold is established using the training set errors:
$$Threshold = \text{mean}(\text{errors}) + 2.5 \times \text{std}(\text{errors})$$
Transactions exceeding this threshold are classified as fraud.

## Performance Results

| Metric | First Autoencoder | Second Autoencoder |
| :--- | :--- | :--- |
| **Precision** | 0.9898 | 0.9927 |
| **Recall** | 0.7926 | 0.8292 |
| **F1-Score** | 0.8803 | 0.9036 |



## Summary
The **Second Autoencoder** outperformed the first model across all metrics, achieving an F1-score of 0.9036. The deeper architecture allowed the model to learn more complex representations of legitimate transactions, leading to a higher recall rate (0.8292) without sacrificing precision.

## Requirements
* TensorFlow / Keras
* Scikit-learn
* Pandas & NumPy
* Matplotlib
