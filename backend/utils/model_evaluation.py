import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation and metrics"""
    
    def __init__(self):
        self.metrics_history = []
    
    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive regression metrics"""
        try:
            metrics = {
                'mae': mean_absolute_error(y_true, y_pred),
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2': r2_score(y_true, y_pred),
                'mape': self._mean_absolute_percentage_error(y_true, y_pred),
                'smape': self._symmetric_mean_absolute_percentage_error(y_true, y_pred)
            }
            
            # Additional metrics
            metrics['max_error'] = np.max(np.abs(y_true - y_pred))
            metrics['mean_error'] = np.mean(y_true - y_pred)
            metrics['std_error'] = np.std(y_true - y_pred)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating regression metrics: {str(e)}")
            raise
    
    def _mean_absolute_percentage_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    def _symmetric_mean_absolute_percentage_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error"""
        return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
    
    def evaluate_classification_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                       threshold: float = 0.5) -> Dict[str, float]:
        """Evaluate classification accuracy for congestion levels"""
        try:
            # Convert continuous predictions to discrete classes
            y_pred_class = self._continuous_to_class(y_pred, threshold)
            y_true_class = self._continuous_to_class(y_true, threshold)
            
            # Calculate accuracy metrics
            accuracy = np.mean(y_true_class == y_pred_class)
            
            # Class-wise accuracy
            classes = np.unique(np.concatenate([y_true_class, y_pred_class]))
            class_accuracy = {}
            
            for cls in classes:
                mask = y_true_class == cls
                if np.sum(mask) > 0:
                    class_accuracy[f'class_{cls}_accuracy'] = np.mean(y_pred_class[mask] == cls)
            
            metrics = {
                'overall_accuracy': accuracy,
                'num_classes': len(classes),
                **class_accuracy
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating classification metrics: {str(e)}")
            raise
    
    def _continuous_to_class(self, values: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Convert continuous values to discrete classes"""
        # Map continuous values to congestion classes
        classes = np.zeros_like(values, dtype=int)
        classes[values <= 1.5] = 1  # low
        classes[(values > 1.5) & (values <= 2.5)] = 2  # medium
        classes[(values > 2.5) & (values <= 3.5)] = 3  # high
        classes[values > 3.5] = 4  # very_high
        
        return classes
    
    def evaluate_time_series_forecast(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    timestamps: List = None) -> Dict[str, any]:
        """Evaluate time series forecasting performance"""
        try:
            # Basic regression metrics
            metrics = self.calculate_regression_metrics(y_true, y_pred)
            
            # Time series specific metrics
            if len(y_true) > 1:
                # Directional accuracy
                true_direction = np.diff(y_true) > 0
                pred_direction = np.diff(y_pred) > 0
                directional_accuracy = np.mean(true_direction == pred_direction)
                metrics['directional_accuracy'] = directional_accuracy
                
                # Trend accuracy
                metrics['trend_correlation'] = np.corrcoef(np.diff(y_true), np.diff(y_pred))[0, 1]
            
            # Peak detection accuracy
            true_peaks = self._detect_peaks(y_true)
            pred_peaks = self._detect_peaks(y_pred)
            
            if len(true_peaks) > 0 and len(pred_peaks) > 0:
                peak_accuracy = len(set(true_peaks) & set(pred_peaks)) / len(set(true_peaks) | set(pred_peaks))
                metrics['peak_detection_accuracy'] = peak_accuracy
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating time series forecast: {str(e)}")
            raise
    
    def _detect_peaks(self, data: np.ndarray, prominence: float = 0.1) -> List[int]:
        """Simple peak detection"""
        peaks = []
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1] and data[i] > np.mean(data) + prominence:
                peaks.append(i)
        return peaks
    
    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray, 
                           cv_folds: int = 5) -> Dict[str, any]:
        """Perform cross-validation evaluation"""
        try:
            from sklearn.model_selection import TimeSeriesSplit
            
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            cv_scores = {
                'mae': [], 'rmse': [], 'r2': [], 'mape': []
            }
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train model (this would need to be adapted for different model types)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                # Calculate metrics
                metrics = self.calculate_regression_metrics(y_val, y_pred)
                for metric in cv_scores:
                    cv_scores[metric].append(metrics[metric])
            
            # Calculate mean and std of CV scores
            cv_results = {}
            for metric in cv_scores:
                cv_results[f'{metric}_mean'] = np.mean(cv_scores[metric])
                cv_results[f'{metric}_std'] = np.std(cv_scores[metric])
            
            return cv_results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            raise
    
    def generate_evaluation_report(self, model_name: str, y_true: np.ndarray, 
                                 y_pred: np.ndarray, model_params: Dict = None) -> Dict[str, any]:
        """Generate comprehensive evaluation report"""
        try:
            report = {
                'model_name': model_name,
                'evaluation_timestamp': pd.Timestamp.now().isoformat(),
                'data_points': len(y_true),
                'model_parameters': model_params or {}
            }
            
            # Regression metrics
            report['regression_metrics'] = self.calculate_regression_metrics(y_true, y_pred)
            
            # Classification metrics
            report['classification_metrics'] = self.evaluate_classification_accuracy(y_true, y_pred)
            
            # Time series metrics
            report['time_series_metrics'] = self.evaluate_time_series_forecast(y_true, y_pred)
            
            # Data distribution analysis
            report['data_analysis'] = {
                'y_true_stats': {
                    'mean': float(np.mean(y_true)),
                    'std': float(np.std(y_true)),
                    'min': float(np.min(y_true)),
                    'max': float(np.max(y_true)),
                    'median': float(np.median(y_true))
                },
                'y_pred_stats': {
                    'mean': float(np.mean(y_pred)),
                    'std': float(np.std(y_pred)),
                    'min': float(np.min(y_pred)),
                    'max': float(np.max(y_pred)),
                    'median': float(np.median(y_pred))
                },
                'residuals_stats': {
                    'mean': float(np.mean(y_true - y_pred)),
                    'std': float(np.std(y_true - y_pred)),
                    'skewness': float(self._calculate_skewness(y_true - y_pred))
                }
            }
            
            # Model performance grade
            report['performance_grade'] = self._calculate_performance_grade(report['regression_metrics'])
            
            # Store in history
            self.metrics_history.append(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating evaluation report: {str(e)}")
            raise
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_performance_grade(self, metrics: Dict[str, float]) -> str:
        """Calculate overall performance grade"""
        try:
            # Normalize metrics (lower is better for MAE, RMSE, MAPE)
            mae_score = max(0, 1 - metrics['mae'] / 2)  # Assuming MAE < 2 is good
            rmse_score = max(0, 1 - metrics['rmse'] / 2)
            r2_score = max(0, metrics['r2'])  # R² should be positive
            mape_score = max(0, 1 - metrics['mape'] / 50)  # MAPE < 50% is reasonable
            
            overall_score = (mae_score + rmse_score + r2_score + mape_score) / 4
            
            if overall_score >= 0.9:
                return 'A+'
            elif overall_score >= 0.8:
                return 'A'
            elif overall_score >= 0.7:
                return 'B+'
            elif overall_score >= 0.6:
                return 'B'
            elif overall_score >= 0.5:
                return 'C+'
            elif overall_score >= 0.4:
                return 'C'
            else:
                return 'D'
                
        except Exception as e:
            logger.error(f"Error calculating performance grade: {str(e)}")
            return 'Unknown'
    
    def compare_models(self, model_reports: List[Dict]) -> Dict[str, any]:
        """Compare multiple model evaluation reports"""
        try:
            if not model_reports:
                return {}
            
            comparison = {
                'models_compared': len(model_reports),
                'comparison_timestamp': pd.Timestamp.now().isoformat(),
                'metrics_comparison': {},
                'best_model': {},
                'rankings': {}
            }
            
            # Extract metrics for comparison
            metrics_to_compare = ['mae', 'rmse', 'r2', 'mape']
            
            for metric in metrics_to_compare:
                values = []
                model_names = []
                
                for report in model_reports:
                    if metric in report.get('regression_metrics', {}):
                        values.append(report['regression_metrics'][metric])
                        model_names.append(report['model_name'])
                
                if values:
                    comparison['metrics_comparison'][metric] = {
                        'values': dict(zip(model_names, values)),
                        'best_model': model_names[np.argmin(values) if metric != 'r2' else np.argmax(values)],
                        'worst_model': model_names[np.argmax(values) if metric != 'r2' else np.argmin(values)]
                    }
            
            # Overall best model based on multiple criteria
            model_scores = {}
            for report in model_reports:
                model_name = report['model_name']
                grade = report.get('performance_grade', 'D')
                
                # Convert grade to numeric score
                grade_scores = {'A+': 10, 'A': 9, 'B+': 8, 'B': 7, 'C+': 6, 'C': 5, 'D': 4}
                model_scores[model_name] = grade_scores.get(grade, 0)
            
            if model_scores:
                best_model_name = max(model_scores, key=model_scores.get)
                comparison['best_model'] = {
                    'name': best_model_name,
                    'score': model_scores[best_model_name],
                    'grade': next(report['performance_grade'] for report in model_reports 
                                if report['model_name'] == best_model_name)
                }
            
            # Rankings
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            comparison['rankings'] = {
                'by_overall_score': [{'model': name, 'score': score} for name, score in sorted_models]
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            raise
    
    def get_metrics_history(self) -> List[Dict]:
        """Get historical metrics for all evaluated models"""
        return self.metrics_history
    
    def export_evaluation_results(self, filepath: str, format: str = 'json'):
        """Export evaluation results to file"""
        try:
            if format == 'json':
                import json
                with open(filepath, 'w') as f:
                    json.dump(self.metrics_history, f, indent=2)
            elif format == 'csv':
                # Flatten the nested structure for CSV export
                flattened_data = []
                for report in self.metrics_history:
                    flat_report = {'model_name': report['model_name']}
                    flat_report.update(report.get('regression_metrics', {}))
                    flat_report.update(report.get('classification_metrics', {}))
                    flattened_data.append(flat_report)
                
                df = pd.DataFrame(flattened_data)
                df.to_csv(filepath, index=False)
            
            logger.info(f"Evaluation results exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting evaluation results: {str(e)}")
            raise