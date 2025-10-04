import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class TrafficDataPreprocessor:
    """Advanced data preprocessing for traffic data"""
    
    def __init__(self):
        self.congestion_mapping = {
            'low': 1, 'medium': 2, 'high': 3, 'very_high': 4
        }
        self.weather_mapping = {
            'clear': 1, 'cloudy': 2, 'rainy': 3, 'foggy': 4, 'stormy': 5
        }
        self.day_mapping = {
            'monday': 1, 'tuesday': 2, 'wednesday': 3, 'thursday': 4,
            'friday': 5, 'saturday': 6, 'sunday': 7
        }
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate traffic data"""
        try:
            # Make a copy to avoid modifying original
            df_clean = df.copy()
            
            # Convert timestamp to datetime
            df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
            
            # Remove duplicates
            df_clean = df_clean.drop_duplicates(subset=['timestamp', 'location'])
            
            # Handle missing values
            df_clean['vehicle_count'] = df_clean['vehicle_count'].fillna(df_clean['vehicle_count'].median())
            df_clean['avg_speed'] = df_clean['avg_speed'].fillna(df_clean['avg_speed'].median())
            
            # Remove outliers using IQR method
            df_clean = self._remove_outliers(df_clean, ['vehicle_count', 'avg_speed'])
            
            # Validate data ranges
            df_clean = df_clean[
                (df_clean['vehicle_count'] >= 0) & 
                (df_clean['vehicle_count'] <= 1000) &
                (df_clean['avg_speed'] >= 0) & 
                (df_clean['avg_speed'] <= 120)
            ]
            
            logger.info(f"Data cleaned: {len(df)} -> {len(df_clean)} records")
            return df_clean
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise
    
    def _remove_outliers(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Remove outliers using IQR method"""
        for col in columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for ML models"""
        try:
            df_features = df.copy()
            
            # Time-based features
            df_features['hour'] = df_features['timestamp'].dt.hour
            df_features['day_of_week_num'] = df_features['timestamp'].dt.dayofweek
            df_features['month'] = df_features['timestamp'].dt.month
            df_features['is_weekend'] = df_features['day_of_week_num'].isin([5, 6]).astype(int)
            df_features['is_rush_hour'] = df_features['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
            
            # Cyclical encoding for time features
            df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
            df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
            df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_week_num'] / 7)
            df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_week_num'] / 7)
            
            # Traffic density features
            df_features['traffic_density'] = df_features['vehicle_count'] / (df_features['avg_speed'] + 1)
            df_features['speed_category'] = pd.cut(
                df_features['avg_speed'], 
                bins=[0, 20, 40, 60, 120], 
                labels=['very_slow', 'slow', 'moderate', 'fast']
            )
            
            # Categorical encoding
            df_features['congestion_numeric'] = df_features['congestion_level'].map(self.congestion_mapping)
            df_features['weather_numeric'] = df_features['weather'].map(self.weather_mapping)
            df_features['day_numeric'] = df_features['day_of_week'].map(self.day_mapping)
            
            # Lag features (previous hour values)
            df_features = df_features.sort_values(['location', 'timestamp'])
            for col in ['vehicle_count', 'avg_speed', 'congestion_numeric']:
                df_features[f'{col}_lag1'] = df_features.groupby('location')[col].shift(1)
                df_features[f'{col}_lag2'] = df_features.groupby('location')[col].shift(2)
            
            # Rolling statistics
            for col in ['vehicle_count', 'avg_speed']:
                df_features[f'{col}_rolling_mean_3h'] = df_features.groupby('location')[col].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
                df_features[f'{col}_rolling_std_3h'] = df_features.groupby('location')[col].rolling(3, min_periods=1).std().reset_index(0, drop=True)
            
            # Fill NaN values created by lag and rolling features
            df_features = df_features.fillna(method='bfill').fillna(method='ffill')
            
            logger.info(f"Feature engineering completed: {len(df_features.columns)} features")
            return df_features
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise
    
    def prepare_sequences(self, df: pd.DataFrame, sequence_length: int = 24, 
                         target_column: str = 'congestion_numeric') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        try:
            # Select feature columns
            feature_columns = [
                'vehicle_count', 'avg_speed', 'congestion_numeric', 'hour',
                'day_of_week_num', 'is_weekend', 'is_rush_hour', 'traffic_density',
                'weather_numeric', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
            ]
            
            # Filter available columns
            available_columns = [col for col in feature_columns if col in df.columns]
            
            if not available_columns:
                raise ValueError("No suitable feature columns found")
            
            # Sort by timestamp
            df_sorted = df.sort_values('timestamp')
            
            # Create sequences
            X, y = [], []
            data = df_sorted[available_columns].values
            target = df_sorted[target_column].values
            
            for i in range(sequence_length, len(data)):
                X.append(data[i-sequence_length:i])
                y.append(target[i])
            
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"Sequences prepared: X shape {X.shape}, y shape {y.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing sequences: {str(e)}")
            raise
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple:
        """Split data into train, validation, and test sets"""
        try:
            n_samples = len(X)
            train_end = int(n_samples * train_ratio)
            val_end = int(n_samples * (train_ratio + val_ratio))
            
            X_train = X[:train_end]
            y_train = y[:train_end]
            X_val = X[train_end:val_end]
            y_val = y[train_end:val_end]
            X_test = X[val_end:]
            y_test = y[val_end:]
            
            logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise
    
    def normalize_features(self, X_train: np.ndarray, X_val: np.ndarray, 
                          X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, object]:
        """Normalize features using training data statistics"""
        try:
            from sklearn.preprocessing import StandardScaler
            
            # Reshape for scaling
            n_samples_train, n_timesteps, n_features = X_train.shape
            X_train_reshaped = X_train.reshape(-1, n_features)
            
            # Fit scaler on training data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_reshaped)
            X_train_scaled = X_train_scaled.reshape(n_samples_train, n_timesteps, n_features)
            
            # Transform validation and test data
            if len(X_val) > 0:
                n_samples_val = X_val.shape[0]
                X_val_reshaped = X_val.reshape(-1, n_features)
                X_val_scaled = scaler.transform(X_val_reshaped)
                X_val_scaled = X_val_scaled.reshape(n_samples_val, n_timesteps, n_features)
            else:
                X_val_scaled = X_val
            
            if len(X_test) > 0:
                n_samples_test = X_test.shape[0]
                X_test_reshaped = X_test.reshape(-1, n_features)
                X_test_scaled = scaler.transform(X_test_reshaped)
                X_test_scaled = X_test_scaled.reshape(n_samples_test, n_timesteps, n_features)
            else:
                X_test_scaled = X_test
            
            logger.info("Feature normalization completed")
            return X_train_scaled, X_val_scaled, X_test_scaled, scaler
            
        except Exception as e:
            logger.error(f"Error normalizing features: {str(e)}")
            raise

# Utility functions
def validate_traffic_data(df: pd.DataFrame) -> Dict[str, any]:
    """Validate traffic data quality"""
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }
    
    try:
        # Check required columns
        required_columns = ['timestamp', 'location', 'vehicle_count', 'avg_speed', 'congestion_level']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
        
        # Check data types
        if 'timestamp' in df.columns:
            try:
                pd.to_datetime(df['timestamp'])
            except:
                validation_results['errors'].append("Invalid timestamp format")
        
        # Check for negative values
        if 'vehicle_count' in df.columns and (df['vehicle_count'] < 0).any():
            validation_results['warnings'].append("Negative vehicle counts found")
        
        if 'avg_speed' in df.columns and (df['avg_speed'] < 0).any():
            validation_results['warnings'].append("Negative speeds found")
        
        # Calculate statistics
        validation_results['statistics'] = {
            'total_records': len(df),
            'unique_locations': df['location'].nunique() if 'location' in df.columns else 0,
            'date_range': {
                'start': df['timestamp'].min() if 'timestamp' in df.columns else None,
                'end': df['timestamp'].max() if 'timestamp' in df.columns else None
            },
            'missing_values': df.isnull().sum().to_dict()
        }
        
    except Exception as e:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"Validation error: {str(e)}")
    
    return validation_results