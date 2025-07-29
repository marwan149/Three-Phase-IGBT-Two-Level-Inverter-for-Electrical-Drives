import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras import models, layers, callbacks
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def alpha_beta_transform(data, phase_a, phase_b, phase_c):
    """Apply Alpha-Beta (Clarke) transformation to three-phase quantities."""
    try:
        sqrt2_3 = np.sqrt(2/3)
        transform_matrix = sqrt2_3 * np.array([
            [1, -0.5, -0.5],
            [0, np.sqrt(3)/2, -np.sqrt(3)/2]
        ])
        phases = np.vstack((data[phase_a], data[phase_b], data[phase_c])).T
        alpha_beta = np.dot(phases, transform_matrix.T)
        return alpha_beta[:, 0], alpha_beta[:, 1]
    except KeyError as e:
        logger.error(f"Missing column in data: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in Alpha-Beta transformation: {e}")
        raise

def park_transform(alpha, beta, theta):
    """Apply Park transformation to convert alpha-beta to d-q components."""
    try:
        theta = np.array(theta)
        if len(theta) != len(alpha) or len(theta) != len(beta):
            raise ValueError("Theta length must match alpha and beta lengths")
        d = np.cos(theta) * alpha + np.sin(theta) * beta
        q = -np.sin(theta) * alpha + np.cos(theta) * beta
        return d, q
    except Exception as e:
        logger.error(f"Error in Park transformation: {e}")
        raise

def plot_clarke_park_transformations(u_alpha, u_beta, u_d, u_q,
                                    i_alpha_k3, i_beta_k3, i_d_k3, i_q_k3,
                                    i_alpha_k2, i_beta_k2, i_d_k2, i_q_k2,
                                    d_alpha_k3, d_beta_k3, d_d_k3, d_q_k3,
                                    d_alpha_k2, d_beta_k2, d_d_k2, d_q_k2,
                                    sample_size):
    """Plot Clarke (α-β) and Park (d-q) transformed components."""
    try:
        fig = plt.figure(figsize=(15, 18))
        
        plt.subplot(5, 2, 1)
        plt.plot(u_alpha[:sample_size], u_beta[:sample_size], 'b.', alpha=0.5)
        plt.title('Clarke Transform: Voltages (α-β)', fontsize=14)
        plt.xlabel('α Voltage', fontsize=12)
        plt.ylabel('β Voltage', fontsize=12)
        plt.grid(True)
        
        plt.subplot(5, 2, 2)
        plt.plot(u_d[:sample_size], u_q[:sample_size], 'b.', alpha=0.5)
        plt.title('Park Transform: Voltages (d-q)', fontsize=14)
        plt.xlabel('d Voltage', fontsize=12)
        plt.ylabel('q Voltage', fontsize=12)
        plt.grid(True)
        
        plt.subplot(5, 2, 3)
        plt.plot(i_alpha_k3[:sample_size], i_beta_k3[:sample_size], 'g.', alpha=0.5)
        plt.title('Clarke Transform: Currents k-3 (α-β)', fontsize=14)
        plt.xlabel('α Current', fontsize=12)
        plt.ylabel('β Current', fontsize=12)
        plt.grid(True)
        
        plt.subplot(5, 2, 4)
        plt.plot(i_d_k3[:sample_size], i_q_k3[:sample_size], 'g.', alpha=0.5)
        plt.title('Park Transform: Currents k-3 (d-q)', fontsize=14)
        plt.xlabel('d Current', fontsize=12)
        plt.ylabel('q Current', fontsize=12)
        plt.grid(True)
        
        plt.subplot(5, 2, 5)
        plt.plot(i_alpha_k2[:sample_size], i_beta_k2[:sample_size], 'g.', alpha=0.5)
        plt.title('Clarke Transform: Currents k-2 (α-β)', fontsize=14)
        plt.xlabel('α Current', fontsize=12)
        plt.ylabel('β Current', fontsize=12)
        plt.grid(True)
        
        plt.subplot(5, 2, 6)
        plt.plot(i_d_k2[:sample_size], i_q_k2[:sample_size], 'g.', alpha=0.5)
        plt.title('Park Transform: Currents k-2 (d-q)', fontsize=14)
        plt.xlabel('d Current', fontsize=12)
        plt.ylabel('q Current', fontsize=12)
        plt.grid(True)
        
        plt.subplot(5, 2, 7)
        plt.plot(d_alpha_k3[:sample_size], d_beta_k3[:sample_size], 'r.', alpha=0.5)
        plt.title('Clarke Transform: Duties k-3 (α-β)', fontsize=14)
        plt.xlabel('α Duty', fontsize=12)
        plt.ylabel('β Duty', fontsize=12)
        plt.grid(True)
        
        plt.subplot(5, 2, 8)
        plt.plot(d_d_k3[:sample_size], d_q_k3[:sample_size], 'r.', alpha=0.5)
        plt.title('Park Transform: Duties k-3 (d-q)', fontsize=14)
        plt.xlabel('d Duty', fontsize=12)
        plt.ylabel('q Duty', fontsize=12)
        plt.grid(True)
        
        plt.subplot(5, 2, 9)
        plt.plot(d_alpha_k2[:sample_size], d_beta_k2[:sample_size], 'r.', alpha=0.5)
        plt.title('Clarke Transform: Duties k-2 (α-β)', fontsize=14)
        plt.xlabel('α Duty', fontsize=12)
        plt.ylabel('β Duty', fontsize=12)
        plt.grid(True)
        
        plt.subplot(5, 2, 10)
        plt.plot(d_d_k2[:sample_size], d_q_k2[:sample_size], 'r.', alpha=0.5)
        plt.title('Park Transform: Duties k-2 (d-q)', fontsize=14)
        plt.xlabel('d Duty', fontsize=12)
        plt.ylabel('q Duty', fontsize=12)
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        return fig
    except Exception as e:
        logger.error(f"Error plotting Clarke and Park transformations: {e}")
        raise

def load_and_prepare_data(file_path, sample_size=5000, outlier_threshold=3, noise_factor=0.01):
    """Load and preprocess inverter dataset with Clarke and Park transformations."""
    try:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError("Loaded dataset is empty")
        
        required_columns = ['u_a_k-1', 'u_b_k-1', 'u_c_k-1', 'i_a_k-3', 'i_b_k-3', 'i_c_k-3',
                           'i_a_k-2', 'i_b_k-2', 'i_c_k-2', 'd_a_k-3', 'd_b_k-3', 'd_c_k-3',
                           'd_a_k-2', 'd_b_k-2', 'd_c_k-2', 'u_dc_k-3', 'u_dc_k-2']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Missing required columns in dataset")
        
        # Handle Missing Values
        if df[required_columns].isnull().any().any():
            logger.warning("Missing values detected. Imputing with median.")
            df[required_columns] = df[required_columns].fillna(df[required_columns].median())
        
        # Outlier Detection and Capping
        for col in required_columns:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df[col] = np.where(z_scores > outlier_threshold, 
                             np.sign(df[col]) * df[col].std() * outlier_threshold + df[col].mean(), 
                             df[col])
        
        # Apply Clarke Transformation
        u_alpha, u_beta = alpha_beta_transform(df, 'u_a_k-1', 'u_b_k-1', 'u_c_k-1')
        i_alpha_k3, i_beta_k3 = alpha_beta_transform(df, 'i_a_k-3', 'i_b_k-3', 'i_c_k-3')
        i_alpha_k2, i_beta_k2 = alpha_beta_transform(df, 'i_a_k-2', 'i_b_k-2', 'i_c_k-2')
        d_alpha_k3, d_beta_k3 = alpha_beta_transform(df, 'd_a_k-3', 'd_b_k-3', 'd_c_k-3')
        d_alpha_k2, d_beta_k2 = alpha_beta_transform(df, 'd_a_k-2', 'd_b_k-2', 'd_c_k-2')
        
        # Estimate theta for Park Transformation
        if 'n_k' in df.columns:
            time_step = 1e-4  # Assume 100us sampling period
            theta = np.cumsum(df['n_k'] * time_step * 2 * np.pi / 60)
        else:
            logger.warning("Speed 'n_k' not found. Assuming 50 Hz frequency for theta.")
            freq = 50
            time_step = 1e-4
            t = np.arange(len(df)) * time_step
            theta = 2 * np.pi * freq * t
        
        # Apply Park Transformation
        u_d, u_q = park_transform(u_alpha, u_beta, theta)
        i_d_k3, i_q_k3 = park_transform(i_alpha_k3, i_beta_k3, theta)
        i_d_k2, i_q_k2 = park_transform(i_alpha_k2, i_beta_k2, theta)
        d_d_k3, d_q_k3 = park_transform(d_alpha_k3, d_beta_k3, theta)
        d_d_k2, d_q_k2 = park_transform(d_alpha_k2, d_beta_k2, theta)
        
        # Plot Clarke and Park Transformations
        plot_clarke_park_transformations(
            u_alpha, u_beta, u_d, u_q,
            i_alpha_k3, i_beta_k3, i_d_k3, i_q_k3,
            i_alpha_k2, i_beta_k2, i_d_k2, i_q_k2,
            d_alpha_k3, d_beta_k3, d_d_k3, d_q_k3,
            d_alpha_k2, d_beta_k2, d_d_k2, d_q_k2,
            sample_size
        )
        
        # Feature Engineering
        features = pd.DataFrame({
            'u_alpha': u_alpha,
            'u_beta': u_beta,
            'u_d': u_d,
            'u_q': u_q,
            'd_alpha_k3': d_alpha_k3,
            'd_beta_k3': d_beta_k3,
            'd_d_k3': d_d_k3,
            'd_q_k3': d_q_k3,
            'i_alpha_k3': i_alpha_k3,
            'i_beta_k3': i_beta_k3,
            'i_d_k3': i_d_k3,
            'i_q_k3': i_q_k3,
            'i_alpha_k2': i_alpha_k2,
            'i_beta_k2': i_beta_k2,
            'i_d_k2': i_d_k2,
            'i_q_k2': i_q_k2,
            'u_dc_k3': df['u_dc_k-3'],
            'u_dc_k2': df['u_dc_k-2'],
            'i_alpha_diff': i_alpha_k2 - i_alpha_k3,
            'i_beta_diff': i_beta_k2 - i_beta_k3,
            'i_d_diff': i_d_k2 - i_d_k3,
            'i_q_diff': i_q_k2 - i_q_k3,
            'u_dc_diff': df['u_dc_k-2'] - df['u_dc_k-3']
        }, index=df.index)  # Ensure features has same index as df
        
        # Feature Selection - Remove highly correlated features
        corr_matrix = features.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
        if to_drop:
            logger.info(f"Dropping highly correlated features: {to_drop}")
            features = features.drop(columns=to_drop)
        
        # Data Augmentation - Add synthetic noise
        noise = np.random.normal(0, noise_factor * features.std(), features.shape)
        features_noisy = features + noise
        features = pd.concat([features, features_noisy], ignore_index=True)
        targets = pd.DataFrame({
            'd_alpha_k2': np.concatenate([d_alpha_k2, d_alpha_k2]),
            'd_beta_k2': np.concatenate([d_beta_k2, d_beta_k2]),
            'd_d_k2': np.concatenate([d_d_k2, d_d_k2]),
            'd_q_k2': np.concatenate([d_q_k2, d_q_k2])
        }, index=features.index)  # Ensure targets has same index as features
        
        # Check Feature Distribution
        logger.info(f"Feature means: {features.mean().to_dict()}")
        logger.info(f"Feature stds: {features.std().to_dict()}")
        
        return features, targets, df, min(sample_size, len(df))
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def plot_three_phases(df, sample_size):
    """Plot time series data for all three phases."""
    try:
        fig = plt.figure(figsize=(20, 12))
        
        plt.subplot(4, 1, 1)
        for phase in ['i_a_k-1', 'i_b_k-1', 'i_c_k-1']:
            df[phase].head(sample_size).plot(label=f'Phase {phase[-5]}')
        plt.title('Phase Currents', fontsize=14)
        plt.ylabel('Current (A)', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.setp(plt.gca().get_xticklabels(), visible=False)
        
        plt.subplot(4, 1, 2)
        for phase in ['u_a_k-1', 'u_b_k-1', 'u_c_k-1']:
            df[phase].head(sample_size).plot(label=f'Phase {phase[-5]}')
        plt.title('Phase Voltages', fontsize=14)
        plt.ylabel('Voltage (V)', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.setp(plt.gca().get_xticklabels(), visible=False)
        
        plt.subplot(4, 1, 3)
        df['u_dc_k-1'].head(sample_size).plot()
        plt.title('DC-link Voltage', fontsize=14)
        plt.ylabel('Voltage (V)', fontsize=12)
        plt.grid(True)
        plt.setp(plt.gca().get_xticklabels(), visible=False)
        
        plt.subplot(4, 1, 4)
        df['n_k'].head(sample_size).plot()
        plt.title('Speed', fontsize=14)
        plt.ylabel('Speed', fontsize=12)
        plt.xlabel('Sample', fontsize=12)
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        return fig
    except Exception as e:
        logger.error(f"Error plotting three phases: {e}")
        raise

def plot_alpha_beta(features, sample_size):
    """Plot Alpha-Beta and d-q components, hexagonal bin plot, and feature distributions."""
    try:
        available_features = features.columns
        has_currents = 'i_alpha_k2' in available_features and 'i_beta_k2' in available_features
        has_duties = 'd_alpha_k3' in available_features and 'd_beta_k3' in available_features
        has_dq = 'i_d_k2' in available_features and 'i_q_k2' in available_features
        has_diffs = 'i_alpha_diff' in available_features and 'i_beta_diff' in available_features and 'u_dc_diff' in available_features
        
        n_subplots = 3 + has_currents + has_duties + has_dq + has_diffs
        fig = plt.figure(figsize=(15, 5 * ((n_subplots + 1) // 2)))
        subplot_idx = 1
        
        plt.subplot((n_subplots + 1) // 2, 2, subplot_idx)
        features['u_alpha'].head(sample_size).plot(label='α')
        features['u_beta'].head(sample_size).plot(label='β')
        plt.title('Alpha-Beta Voltages', fontsize=14)
        plt.ylabel('Voltage', fontsize=12)
        plt.legend()
        plt.grid(True)
        subplot_idx += 1
        
        if has_dq:
            plt.subplot((n_subplots + 1) // 2, 2, subplot_idx)
            features['u_d'].head(sample_size).plot(label='d')
            features['u_q'].head(sample_size).plot(label='q')
            plt.title('d-q Voltages', fontsize=14)
            plt.ylabel('Voltage', fontsize=12)
            plt.legend()
            plt.grid(True)
            subplot_idx += 1
        
        if has_currents:
            plt.subplot((n_subplots + 1) // 2, 2, subplot_idx)
            features['i_alpha_k2'].head(sample_size).plot(label='α')
            features['i_beta_k2'].head(sample_size).plot(label='β')
            plt.title('Alpha-Beta Currents (k-2)', fontsize=14)
            plt.ylabel('Current', fontsize=12)
            plt.legend()
            plt.grid(True)
            subplot_idx += 1
        
        if has_dq:
            plt.subplot((n_subplots + 1) // 2, 2, subplot_idx)
            features['i_d_k2'].head(sample_size).plot(label='d')
            features['i_q_k2'].head(sample_size).plot(label='q')
            plt.title('d-q Currents (k-2)', fontsize=14)
            plt.ylabel('Current', fontsize=12)
            plt.legend()
            plt.grid(True)
            subplot_idx += 1
        
        if has_duties:
            plt.subplot((n_subplots + 1) // 2, 2, subplot_idx)
            features['d_alpha_k3'].head(sample_size).plot(label='α')
            features['d_beta_k3'].head(sample_size).plot(label='β')
            plt.title('Alpha-Beta Duties (k-3)', fontsize=14)
            plt.ylabel('Duty', fontsize=12)
            plt.legend()
            plt.grid(True)
            subplot_idx += 1
        
        if has_dq:
            plt.subplot((n_subplots + 1) // 2, 2, subplot_idx)
            features['d_d_k3'].head(sample_size).plot(label='d')
            features['d_q_k3'].head(sample_size).plot(label='q')
            plt.title('d-q Duties (k-3)', fontsize=14)
            plt.ylabel('Duty', fontsize=12)
            plt.legend()
            plt.grid(True)
            subplot_idx += 1
        
        plt.subplot((n_subplots + 1) // 2, 2, subplot_idx)
        plt.hexbin(features['u_alpha'].head(sample_size), 
                  features['u_beta'].head(sample_size),
                  gridsize=50, cmap='viridis', mincnt=1)
        plt.colorbar(label='Count')
        plt.title('Hexagonal Bin Plot: Alpha vs Beta Voltages', fontsize=14)
        plt.xlabel('α Voltage', fontsize=12)
        plt.ylabel('β Voltage', fontsize=12)
        plt.grid(True)
        subplot_idx += 1
        
        if has_diffs:
            plt.subplot((n_subplots + 1) // 2, 2, subplot_idx)
            sns.histplot(features['i_alpha_diff'].head(sample_size), kde=True, label='i_alpha_diff')
            sns.histplot(features['i_beta_diff'].head(sample_size), kde=True, label='i_beta_diff')
            sns.histplot(features['i_d_diff'].head(sample_size), kde=True, label='i_d_diff')
            sns.histplot(features['i_q_diff'].head(sample_size), kde=True, label='i_q_diff')
            plt.title('Distribution of Current Differences', fontsize=14)
            plt.xlabel('Value', fontsize=12)
            plt.legend()
            plt.grid(True)
            subplot_idx += 1
            
            plt.subplot((n_subplots + 1) // 2, 2, subplot_idx)
            sns.histplot(features['u_dc_diff'].head(sample_size), kde=True, label='u_dc_diff')
            plt.title('Distribution of DC Voltage Difference', fontsize=14)
            plt.xlabel('Value', fontsize=12)
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        return fig
    except Exception as e:
        logger.error(f"Error plotting Alpha-Beta and d-q visualizations: {e}")
        raise

def plot_correlation_heatmap(features):
    """Plot correlation heatmap of features."""
    try:
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(features.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap', fontsize=14)
        plt.tight_layout()
        plt.show()
        return fig
    except Exception as e:
        logger.error(f"Error plotting correlation heatmap: {e}")
        raise

def plot_predictions(model, X_test, y_test, sample_size=100):
    """Plot model predictions vs actual values with smoothing."""
    try:
        predictions = model.predict(X_test[:sample_size], verbose=0)
        
        window_size = 5
        actual_smooth_alpha = pd.Series(y_test['d_alpha_k2'][:sample_size].values).rolling(window=window_size, center=True).mean()
        pred_smooth_alpha = pd.Series(predictions[:, 0].flatten()).rolling(window=window_size, center=True).mean()
        actual_smooth_d = pd.Series(y_test['d_d_k2'][:sample_size].values).rolling(window=window_size, center=True).mean()
        pred_smooth_d = pd.Series(predictions[:, 1].flatten()).rolling(window=window_size, center=True).mean()
        
        actual_smooth_alpha = actual_smooth_alpha.fillna(method='bfill').fillna(method='ffill')
        pred_smooth_alpha = pred_smooth_alpha.fillna(method='bfill').fillna(method='ffill')
        actual_smooth_d = actual_smooth_d.fillna(method='bfill').fillna(method='ffill')
        pred_smooth_d = pred_smooth_d.fillna(method='bfill').fillna(method='ffill')
        
        fig = plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(actual_smooth_alpha, 'b-', label='Actual α')
        plt.plot(pred_smooth_alpha, 'r--', label='Predicted α')
        plt.title('Model Predictions vs Actual (Alpha Component)', fontsize=14)
        plt.xlabel('Sample', fontsize=12)
        plt.ylabel('Duty Alpha', fontsize=12)
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(actual_smooth_d, 'b-', label='Actual d')
        plt.plot(pred_smooth_d, 'r--', label='Predicted d')
        plt.title('Model Predictions vs Actual (d Component)', fontsize=14)
        plt.xlabel('Sample', fontsize=12)
        plt.ylabel('Duty d', fontsize=12)
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        return fig
    except Exception as e:
        logger.error(f"Error plotting predictions: {e}")
        raise

def create_neural_networks(input_dim):
    """Create and compile neural network models with dropout."""
    try:
        robust_model = models.Sequential([
            layers.Dense(units=30, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.2),
            layers.Dense(units=30, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(units=30, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(units=30, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(units=30, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(units=30, activation='relu'),
            layers.Dense(units=2)  # Output d_alpha_k2 and d_d_k2
        ])
        robust_model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        return robust_model
    except Exception as e:
        logger.error(f"Error creating neural networks: {e}")
        raise

def create_pipeline():
    """Create a scikit-learn pipeline for preprocessing."""
    try:
        return Pipeline([('scaler', StandardScaler())])
    except Exception as e:
        logger.error(f"Error creating pipeline: {e}")
        raise

def train_and_evaluate(model, X_train, y_train, X_test, y_test, epochs=30, batch_size=300):
    """Train model with early stopping and return training history."""
    try:
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=1
        )
        return history
    except Exception as e:
        logger.error(f"Error in training: {e}")
        raise

def cross_validate_model(model, X, y, n_splits=5, epochs=30, batch_size=300):
    """Perform k-fold cross-validation."""
    try:
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
            logger.info(f"Training fold {fold}/{n_splits}")
            X_train, X_val = X[train_idx], X[val_idx]
            # Use iloc to select rows by integer indices
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            pipeline = create_pipeline()
            pipeline.fit(X_train)
            X_train_scaled = pipeline.transform(X_train)
            X_val_scaled = pipeline.transform(X_val)
            
            history = train_and_evaluate(model, X_train_scaled, y_train, 
                                      X_val_scaled, y_val, epochs, batch_size)
            cv_scores.append(min(history.history['val_loss']))
        
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        return cv_scores
    except Exception as e:
        logger.error(f"Error in cross-validation: {e}")
        raise

def plot_training_history(history):
    """Plot training and validation loss."""
    try:
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(train_loss) + 1)
        
        fig = plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, 'r--', label='Training Loss')
        plt.plot(epochs, val_loss, 'b-', label='Validation Loss')
        plt.title('Model Training History (Alpha-d)', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.show()
        return fig
    except Exception as e:
        logger.error(f"Error plotting training history: {e}")
        raise

def main():
    try:
        # Load and prepare data
        file_path = '/kaggle/input/inverter-data-set/Inverter Data Set.csv'
        X, y, df, sample_size = load_and_prepare_data(file_path, sample_size=5000, 
                                                    outlier_threshold=3, noise_factor=0.01)
        logger.info("Data loaded and preprocessed successfully")
        
        # Visualize data
        plot_three_phases(df, sample_size)
        plot_alpha_beta(X, sample_size)
        plot_correlation_heatmap(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y[['d_alpha_k2', 'd_d_k2']], test_size=0.3, random_state=42
        )
        
        # Print shapes
        logger.info(f"Training features shape: {X_train.shape}")
        logger.info(f"Testing features shape: {X_test.shape}")
        logger.info(f"Training targets shape: {y_train.shape}")
        logger.info(f"Testing targets shape: {y_test.shape}")
        
        # Create pipeline
        pipeline = create_pipeline()
        pipeline.fit(X_train)
        X_train_scaled = pipeline.transform(X_train)
        X_test_scaled = pipeline.transform(X_test)
        
        # Print standardization statistics
        n_features = X_train_scaled.shape[1]
        if n_features > 0:
            logger.info(f"Mean of standardized test dataset (feature 0): {round(X_test_scaled[:, 0].mean())}")
            if n_features > 1:
                logger.info(f"Std of standardized training dataset (feature 1): {round(X_train_scaled[:, 1].std())}")
        else:
            logger.warning("No features available after preprocessing")
        
        # Create and train model
        robust_model = create_neural_networks(input_dim=X_train.shape[1])
        cv_scores = cross_validate_model(robust_model, pipeline.transform(X), y[['d_alpha_k2', 'd_d_k2']])
        
        history = train_and_evaluate(
            robust_model,
            X_train_scaled,
            y_train[['d_alpha_k2', 'd_d_k2']],
            X_test_scaled,
            y_test[['d_alpha_k2', 'd_d_k2']]
        )
        
        # Plot results
        plot_training_history(history)
        plot_predictions(robust_model, X_test_scaled, y_test)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()