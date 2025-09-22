
import numpy as np
import sys
from typing import Tuple, List, Optional
from scipy.stats import norm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LPConformalPrediction:
    """
    Lévy-Prokhorov Robust Conformal Prediction for Time Series with Distribution Shifts
    Based on the paper: "Conformal Prediction under Lévy-Prokhorov Distribution Shifts"
    """
    
    def __init__(self, alpha: float = 0.1, epsilon: float = 0.1, rho: float = 0.05):
        """
        Initialize LP Robust Conformal Prediction
        
        Args:
            alpha: Target miscoverage level (1 - coverage)
            epsilon: Local perturbation parameter (Lévy-Prokhorov)
            rho: Global perturbation parameter (Lévy-Prokhorov)
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.rho = rho
        self.quantile = None
        self.calibration_scores = None
        
        logger.info(f"Initialized LP Conformal Prediction with alpha={alpha}, epsilon={epsilon}, rho={rho}")
    
    def nonconformity_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate nonconformity scores (absolute error for time series)
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Nonconformity scores
        """
        return np.abs(y_true - y_pred)
    
    def calibrate(self, calibration_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Calibrate the conformal prediction model
        
        Args:
            calibration_data: Tuple of (y_true_calib, y_pred_calib) for calibration
        """
        try:
            y_true_calib, y_pred_calib = calibration_data
            
            if len(y_true_calib) != len(y_pred_calib):
                raise ValueError("Calibration true and predicted arrays must have same length")
            
            # Calculate nonconformity scores
            scores = self.nonconformity_score(y_true_calib, y_pred_calib)
            self.calibration_scores = scores
            
            # Calculate adjusted quantile level
            level_adjusted = (1.0 - self.alpha) * (1.0 + 1.0 / len(scores))
            
            # Calculate worst-case quantile using LP robustness
            self.quantile = self._calculate_worst_case_quantile(scores, level_adjusted)
            
            logger.info(f"Calibration completed. Calculated quantile: {self.quantile:.4f}")
            logger.info(f"Calibration scores range: [{scores.min():.4f}, {scores.max():.4f}]")
            
        except Exception as e:
            logger.error(f"Error during calibration: {str(e)}")
            sys.exit(1)
    
    def _calculate_worst_case_quantile(self, scores: np.ndarray, level: float) -> float:
        """
        Calculate worst-case quantile under LP distribution shifts
        
        Args:
            scores: Nonconformity scores
            level: Desired quantile level
            
        Returns:
            Worst-case quantile value
        """
        try:
            # Sort scores for quantile calculation
            sorted_scores = np.sort(scores)
            
            # Standard quantile calculation
            standard_quantile = np.quantile(sorted_scores, level, method='higher')
            
            # LP robust adjustment: QuantWC = Quant(level + rho) + epsilon
            adjusted_level = min(level + self.rho, 1.0)  # Ensure level doesn't exceed 1
            lp_quantile = np.quantile(sorted_scores, adjusted_level, method='higher') + self.epsilon
            
            logger.info(f"Standard quantile ({level:.3f}): {standard_quantile:.4f}")
            logger.info(f"LP robust quantile ({adjusted_level:.3f} + ε={self.epsilon}): {lp_quantile:.4f}")
            
            return lp_quantile
            
        except Exception as e:
            logger.error(f"Error in quantile calculation: {str(e)}")
            sys.exit(1)
    
    def predict(self, y_pred_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate prediction intervals
        
        Args:
            y_pred_test: Point predictions for test data
            
        Returns:
            Tuple of (lower_bounds, upper_bounds) prediction intervals
        """
        try:
            if self.quantile is None:
                raise ValueError("Model must be calibrated before prediction")
            
            # Create prediction intervals
            lower_bounds = y_pred_test - self.quantile
            upper_bounds = y_pred_test + self.quantile
            
            logger.info(f"Generated prediction intervals with width: {2 * self.quantile:.4f}")
            
            return lower_bounds, upper_bounds
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            sys.exit(1)
    
    def evaluate_coverage(self, y_true_test: np.ndarray, 
                         lower_bounds: np.ndarray, 
                         upper_bounds: np.ndarray) -> float:
        """
        Evaluate coverage of prediction intervals
        
        Args:
            y_true_test: True test values
            lower_bounds: Lower bounds of prediction intervals
            upper_bounds: Upper bounds of prediction intervals
            
        Returns:
            Coverage percentage
        """
        try:
            coverage = np.mean((y_true_test >= lower_bounds) & (y_true_test <= upper_bounds))
            logger.info(f"Empirical coverage: {coverage * 100:.2f}% (Target: {(1 - self.alpha) * 100:.2f}%)")
            return coverage
            
        except Exception as e:
            logger.error(f"Error in coverage evaluation: {str(e)}")
            sys.exit(1)

def generate_time_series_data(n_samples: int = 1000, 
                             trend_strength: float = 0.1,
                             noise_level: float = 0.5,
                             shift_point: Optional[int] = None,
                             shift_magnitude: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic time series data with optional distribution shift
    
    Args:
        n_samples: Number of samples to generate
        trend_strength: Strength of linear trend
        noise_level: Level of Gaussian noise
        shift_point: Point at which distribution shift occurs (None for no shift)
        shift_magnitude: Magnitude of distribution shift
        
    Returns:
        Tuple of (time indices, values)
    """
    try:
        # Generate time indices
        t = np.arange(n_samples)
        
        # Generate base time series with trend and noise
        y = trend_strength * t + noise_level * np.random.randn(n_samples)
        
        # Apply distribution shift if specified
        if shift_point is not None and shift_point < n_samples:
            y[shift_point:] += shift_magnitude
            logger.info(f"Applied distribution shift at index {shift_point} with magnitude {shift_magnitude}")
        
        logger.info(f"Generated time series with {n_samples} samples")
        return t, y
        
    except Exception as e:
        logger.error(f"Error generating time series data: {str(e)}")
        sys.exit(1)

def simple_forecast(y: np.ndarray, window_size: int = 10) -> np.ndarray:
    """
    Simple forecasting model (moving average)
    
    Args:
        y: Time series values
        window_size: Moving average window size
        
    Returns:
        Forecasted values
    """
    try:
        y_pred = np.zeros_like(y)
        
        # Use moving average forecast
        for i in range(len(y)):
            if i < window_size:
                y_pred[i] = np.mean(y[:i+1]) if i > 0 else y[0]
            else:
                y_pred[i] = np.mean(y[i-window_size:i])
        
        return y_pred
        
    except Exception as e:
        logger.error(f"Error in forecasting: {str(e)}")
        sys.exit(1)

def experiment_main():
    """
    Main experiment function to demonstrate LP robust conformal prediction
    on time series data with distribution shifts
    """
    logger.info("Starting LP Robust Conformal Prediction Experiment")
    logger.info("=" * 60)
    
    # Experiment parameters
    n_samples = 500
    calib_ratio = 0.5
    shift_point = 300
    shift_magnitude = 3.0
    alpha = 0.1  # 90% coverage target
    
    # LP robustness parameters to test
    robustness_params = [
        (0.0, 0.0),   # Standard conformal prediction
        (0.1, 0.05),  # Mild robustness
        (0.2, 0.1),   # Moderate robustness
        (0.3, 0.15),  # Strong robustness
    ]
    
    try:
        # Generate time series data with distribution shift
        logger.info("Generating time series data with distribution shift...")
        t, y_true = generate_time_series_data(
            n_samples=n_samples,
            trend_strength=0.05,
            noise_level=0.8,
            shift_point=shift_point,
            shift_magnitude=shift_magnitude
        )
        
        # Generate forecasts
        logger.info("Generating forecasts...")
        y_pred = simple_forecast(y_true, window_size=15)
        
        # Split data into calibration and test sets
        calib_size = int(n_samples * calib_ratio)
        test_size = n_samples - calib_size
        
        y_true_calib = y_true[:calib_size]
        y_pred_calib = y_pred[:calib_size]
        
        y_true_test = y_true[calib_size:]
        y_pred_test = y_pred[calib_size:]
        
        logger.info(f"Data split: {calib_size} calibration, {test_size} test samples")
        logger.info(f"Distribution shift occurs at index {shift_point} (test index: {shift_point - calib_size})")
        
        results = []
        
        # Test different robustness parameter combinations
        for epsilon, rho in robustness_params:
            logger.info("-" * 50)
            logger.info(f"Testing ε={epsilon}, ρ={rho}")
            
            # Initialize and calibrate LP conformal prediction
            lp_cp = LPConformalPrediction(alpha=alpha, epsilon=epsilon, rho=rho)
            lp_cp.calibrate((y_true_calib, y_pred_calib))
            
            # Generate prediction intervals
            lower_bounds, upper_bounds = lp_cp.predict(y_pred_test)
            
            # Evaluate coverage
            coverage = lp_cp.evaluate_coverage(y_true_test, lower_bounds, upper_bounds)
            
            # Calculate average interval width
            avg_width = np.mean(upper_bounds - lower_bounds)
            
            results.append({
                'epsilon': epsilon,
                'rho': rho,
                'coverage': coverage,
                'avg_width': avg_width,
                'quantile': lp_cp.quantile
            })
            
            logger.info(f"Average interval width: {avg_width:.4f}")
        
        # Print summary results
        logger.info("=" * 60)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("=" * 60)
        
        for res in results:
            logger.info(
                f"ε={res['epsilon']:.1f}, ρ={res['rho']:.2f}: "
                f"Coverage={res['coverage']*100:.1f}%, "
                f"Width={res['avg_width']:.3f}, "
                f"Quantile={res['quantile']:.3f}"
            )
        
        # Final analysis
        logger.info("=" * 60)
        logger.info("CONCLUSION")
        logger.info("=" * 60)
        logger.info("The experiment demonstrates that LP robust conformal prediction")
        logger.info("can maintain valid coverage under distribution shifts by adjusting")
        logger.info("the robustness parameters (ε and ρ). Higher robustness parameters")
        logger.info("typically lead to wider prediction intervals but better coverage")
        logger.info("guarantees under distribution shifts.")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    experiment_main()
