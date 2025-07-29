# Three-Phase-IGBT-Two-Level-Inverter-for-Electrical-Drives


Overview
This project implements a comprehensive pipeline for analyzing and modeling inverter data using Python. It performs Clarke and Park transformations on three-phase electrical quantities, preprocesses data, visualizes key components, and trains a neural network to predict duty cycles.

Features

Data Preprocessing: Loads inverter dataset, handles missing values, detects and caps outliers, and applies data augmentation with synthetic noise.
Transformations: Implements Clarke (α-β) and Park (d-q) transformations for voltages, currents, and duty cycles.
Visualization: Generates plots for:
Three-phase time series (currents, voltages, DC-link voltage, speed)
Clarke and Park transformed components
Alpha-Beta and d-q component time series
Feature correlation heatmap
Model predictions vs. actual values


Neural Network: Builds and trains a deep neural network with dropout for predicting duty cycle components (d_alpha_k2, d_d_k2).
Cross-Validation: Performs k-fold cross-validation to evaluate model performance.
Pipeline: Uses scikit-learn's Pipeline for feature scaling.

Requirements

Python 3.8+
Libraries:
pandas
numpy
matplotlib
seaborn
scikit-learn
keras


Dataset: Inverter Data Set (Inverter Data Set.csv) with required columns:
u_a_k-1, u_b_k-1, u_c_k-1 (phase voltages)
i_a_k-3, i_b_k-3, i_c_k-3, i_a_k-2, i_b_k-2, i_c_k-2 (phase currents)
d_a_k-3, d_b_k-3, d_c_k-3, d_a_k-2, d_b_k-2, d_c_k-2 (duty cycles)
u_dc_k-3, u_dc_k-2 (DC-link voltages)
n_k (speed, optional)



Installation

Clone the repository:git clone <repository-url>
cd <repository-directory>


Install dependencies:pip install -r requirements.txt


Place the dataset (Inverter Data Set.csv) in the project directory or update the file path in the main() function.

Usage
Run the main script to execute the full pipeline:
python Model.py

The script will:

Load and preprocess the dataset.
Apply Clarke and Park transformations.
Generate visualizations.
Train and evaluate a neural network model.
Perform k-fold cross-validation.
Save plots and log results.

File Structure

Model.py: Main script containing all functions and the main execution logic.
Inverter Data Set.csv: Input dataset (not included, must be provided).
README.md: This file.

Key Functions

alpha_beta_transform: Applies Clarke transformation to three-phase quantities.
park_transform: Converts α-β components to d-q components.
load_and_prepare_data: Loads data, handles preprocessing, and applies transformations.
plot_clarke_park_transformations: Visualizes Clarke and Park transformed components.
plot_three_phases: Plots time series of three-phase quantities.
plot_alpha_beta: Plots α-β and d-q components with hexagonal bin plots.
plot_correlation_heatmap: Generates a feature correlation heatmap.
create_neural_networks: Builds the neural network model.
train_and_evaluate: Trains the model with early stopping.
cross_validate_model: Performs k-fold cross-validation.
main: Orchestrates the entire pipeline.

Output

Logs: Console output with preprocessing details, model training progress, and cross-validation scores.
Plots:
Time series of phase currents, voltages, DC-link voltage, and speed.
Clarke and Park transformation scatter plots.
Alpha-Beta and d-q component time series.
Feature correlation heatmap.
Training history (loss curves).
Model predictions vs. actual values.



Notes

The dataset path in main() assumes the file is located at /kaggle/input/inverter-data-set/Inverter Data Set.csv. Update this path as needed.
The neural network is configured with dropout layers to prevent overfitting.
Cross-validation uses 5 folds by default, adjustable via the n_splits parameter.
Sample size for visualizations is set to 5000 by default to manage memory usage.

License
This project is licensed under the MIT License.
