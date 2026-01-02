# Credit Card Fraud Detection Using Deep Generative Models

This project implements an anomaly detection system for credit card fraud using **Autoencoders (AE)** and **Variational Autoencoders (VAE)**. By training these models exclusively on non-fraudulent transactions, they learn to reconstruct "normal" patterns. When a fraudulent transaction occurs, the resulting high reconstruction error allows the system to flag it as an anomaly.

## Methodology

### 1. Data Preprocessing
* **Feature Engineering**: The `Time` and `Amount` features are log-scaled to reduce the impact of outliers.
* **Normalization**: All features are scaled to a range of (0, 1) using `MinMaxScaler`.
* **Data Splitting**: Training and validation sets consist entirely of non-fraudulent transactions. The test set is a combination of remaining non-fraud and all available fraud transactions.

### 2. Model Architectures
The project explores two primary deep learning approaches:

#### Standard Autoencoder (AE)
* **First AE**: A shallow model consisting of an input layer, one encoding layer with 14 units, and a decoding layer.
* **Second AE**: A deeper model featuring two encoding layers (14 and 10 units) and two decoding layers.



[Image of Autoencoder architecture]


#### Variational Autoencoder (VAE)
The VAE maps input data to a latent space distribution rather than fixed points, regularizing the latent space via Kullback-Leibler (KL) divergence.
* **First VAE**: Features an encoder with a 64-unit hidden layer mapping to a 14-dimensional latent space.
* **Second VAE**: A variation featuring smaller hidden layers (20 and 12 units) to explore the impact of model capacity on performance.



### 3. Fraud Detection Mechanism
The models classify transactions based on **Reconstruction Error** (Mean Squared Error). A threshold is established based on the training set errors:
* **Standard AE Threshold**: $\text{mean}(\text{errors}) + 2.5 \times \text{std}(\text{errors})$.
* **VAE Threshold**: $\text{mean}(\text{errors}) + 1.5 \times \text{std}(\text{errors})$.

Transactions exceeding this threshold are classified as fraud.

## Performance Results

| Model Configuration | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **First Autoencoder** | 0.9898 | 0.7926 | 0.8803 |
| **Second Autoencoder** | **0.9927** | 0.8292 | **0.9036** |
| **First VAE** | 0.9691 | **0.8292** | 0.8937 |
| **Second VAE** | 0.9691 | **0.8292** | 0.8937 |



## Summary
* **Standard Autoencoder**: The **Second AE** achieved the highest overall F1-score of 0.9036. Its deeper architecture allowed for more complex representations of legitimate transactions.
* **Variational Autoencoder**: Both VAE configurations achieved a consistent Recall of 0.8292. While the VAE precision was slightly lower than the Standard AE, it provided stable performance across different architectures.

## Requirements
* TensorFlow / Keras
* Scikit-learn
* Pandas & NumPy
* Matplotlib
