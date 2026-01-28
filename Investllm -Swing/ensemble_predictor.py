"""
Ensemble Predictor with Temperature Scaling for V3 Strategy

Features:
1. Temperature scaling for confidence calibration
2. Ensemble of multiple models (average predictions)
3. Confidence based on model agreement
4. GPU-accelerated inference
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pickle


class TemperatureScaler:
    """
    Temperature scaling for model calibration

    If T < 1: Sharpens predictions (underconfident model -> more decisive)
    If T > 1: Softens predictions (overconfident model -> more conservative)
    """

    def __init__(self, temperature: float = 0.5):
        self.temperature = temperature

    def scale(self, predictions: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to predictions"""
        # Clip to avoid log(0)
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)

        # Convert to logits
        logits = np.log(predictions / (1 - predictions))

        # Scale by temperature
        scaled_logits = logits / self.temperature

        # Convert back to probabilities
        calibrated = 1 / (1 + np.exp(-scaled_logits))

        return calibrated

    def find_optimal_temperature(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        temperature_range: List[float] = None
    ) -> float:
        """
        Find optimal temperature that minimizes calibration error

        Args:
            predictions: Model predictions (0-1)
            actuals: Actual outcomes (0 or 1)
            temperature_range: List of temperatures to try

        Returns:
            Optimal temperature value
        """
        if temperature_range is None:
            temperature_range = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]

        best_temp = 1.0
        best_ece = float('inf')

        for temp in temperature_range:
            self.temperature = temp
            calibrated = self.scale(predictions)
            ece = self._calculate_ece(calibrated, actuals)

            if ece < best_ece:
                best_ece = ece
                best_temp = temp

        self.temperature = best_temp
        return best_temp

    def _calculate_ece(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error

        A well-calibrated model has ECE close to 0
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            in_bin = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i + 1])
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                avg_confidence = np.mean(predictions[in_bin])
                avg_accuracy = np.mean(actuals[in_bin])
                ece += prop_in_bin * abs(avg_confidence - avg_accuracy)

        return ece


class EnsemblePredictor:
    """
    Ensemble of multiple LSTM models with temperature scaling

    Benefits:
    - Reduces noise from single model
    - Agreement = high confidence (genuine signal)
    - Disagreement = low confidence (uncertain)
    """

    def __init__(
        self,
        model_paths: List[str] = None,
        model_class: nn.Module = None,
        temperature: float = 0.5,
        device: str = 'cpu'
    ):
        self.model_paths = model_paths or []
        self.model_class = model_class
        self.temperature_scaler = TemperatureScaler(temperature)
        self.device = device
        self.models = []
        self.scalers = []
        self.loaded = False

    def load_models(self, input_dim: int = None):
        """Load all ensemble models"""
        if not self.model_paths:
            print("No model paths provided")
            return False

        self.models = []
        self.scalers = []

        for path in self.model_paths:
            model_path = Path(path)
            if not model_path.exists():
                print(f"Model not found: {path}")
                continue

            try:
                # Load checkpoint
                checkpoint = torch.load(model_path, map_location=self.device)

                # Create model
                if self.model_class is not None:
                    model = self.model_class(input_dim=input_dim)
                else:
                    # Try to infer from checkpoint
                    from train_swing_model_FIXED import SwingLSTM
                    model = SwingLSTM(input_dim=input_dim)

                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(self.device)
                model.eval()
                self.models.append(model)

                # Load scaler if available
                scaler_path = model_path.parent / 'scaler.pkl'
                if scaler_path.exists():
                    with open(scaler_path, 'rb') as f:
                        self.scalers.append(pickle.load(f))

                print(f"Loaded model: {path}")

            except Exception as e:
                print(f"Error loading {path}: {e}")

        self.loaded = len(self.models) > 0
        print(f"\nEnsemble loaded: {len(self.models)} models")
        return self.loaded

    def predict(
        self,
        features: np.ndarray,
        return_individual: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make ensemble prediction

        Args:
            features: Input features (batch_size, sequence_length, num_features)
            return_individual: If True, also return individual model predictions

        Returns:
            (calibrated_predictions, confidence_scores)

            confidence_scores are based on:
            - Temperature-scaled prediction value
            - Model agreement (low std = high confidence)
        """
        if not self.loaded or len(self.models) == 0:
            raise ValueError("No models loaded. Call load_models() first.")

        # Convert to tensor
        if isinstance(features, np.ndarray):
            features = torch.FloatTensor(features).to(self.device)

        # Get predictions from all models
        all_predictions = []

        with torch.no_grad():
            for model in self.models:
                pred = model(features)
                # Convert to probability if needed
                if pred.dim() > 1:
                    pred = pred.squeeze()
                # Ensure in 0-1 range
                pred = torch.sigmoid(pred) if pred.min() < 0 or pred.max() > 1 else pred
                all_predictions.append(pred.cpu().numpy())

        # Stack predictions
        all_predictions = np.array(all_predictions)  # (n_models, batch_size)

        # Average predictions
        mean_predictions = np.mean(all_predictions, axis=0)

        # Calculate model agreement (std)
        std_predictions = np.std(all_predictions, axis=0)

        # Apply temperature scaling
        calibrated = self.temperature_scaler.scale(mean_predictions)

        # Calculate confidence score
        # High confidence if: high prediction AND models agree
        agreement_bonus = 1 - np.clip(std_predictions * 5, 0, 0.3)  # Max 30% bonus

        # Base confidence from calibrated prediction distance from 0.5
        base_confidence = np.abs(calibrated - 0.5) * 2  # 0-1 scale

        # Combined confidence
        confidence = np.clip(base_confidence * agreement_bonus + 0.3, 0, 1)

        if return_individual:
            return calibrated, confidence, all_predictions

        return calibrated, confidence

    def predict_single(self, features: np.ndarray) -> Dict:
        """
        Predict for a single sample with detailed output

        Returns:
            {
                'prediction': float,  # 0-1 probability of going up
                'confidence': float,  # 0-1 confidence score
                'direction': str,     # 'UP' or 'DOWN'
                'confidence_level': str,  # 'HIGH', 'MEDIUM', 'LOW'
                'model_agreement': float  # 0-1, higher = more agreement
            }
        """
        if features.ndim == 2:
            features = features[np.newaxis, ...]  # Add batch dimension

        calibrated, confidence, individual = self.predict(features, return_individual=True)

        pred = calibrated[0]
        conf = confidence[0]
        std = np.std(individual[:, 0])

        return {
            'prediction': float(pred),
            'confidence': float(conf),
            'direction': 'UP' if pred > 0.5 else 'DOWN',
            'confidence_level': 'HIGH' if conf >= 0.8 else 'MEDIUM' if conf >= 0.6 else 'LOW',
            'model_agreement': float(1 - min(std * 5, 1)),
            'individual_predictions': individual[:, 0].tolist()
        }

    def calibrate(
        self,
        val_features: np.ndarray,
        val_targets: np.ndarray
    ) -> float:
        """
        Calibrate temperature using validation data

        Returns:
            Optimal temperature
        """
        print("Calibrating ensemble temperature...")

        # Get raw predictions
        calibrated, _ = self.predict(val_features)

        # Convert targets to binary (up/down)
        binary_targets = (val_targets > 0).astype(float)

        # Find optimal temperature
        # Temporarily set temperature to 1.0 to get uncalibrated predictions
        original_temp = self.temperature_scaler.temperature
        self.temperature_scaler.temperature = 1.0
        raw_pred, _ = self.predict(val_features)

        optimal_temp = self.temperature_scaler.find_optimal_temperature(
            raw_pred, binary_targets
        )

        print(f"Optimal temperature: {optimal_temp}")
        print(f"(Original was: {original_temp})")

        return optimal_temp


class SingleModelPredictor(EnsemblePredictor):
    """
    Single model predictor with temperature scaling
    (For when only one model is available)
    """

    def __init__(
        self,
        model_path: str,
        model_class: nn.Module = None,
        temperature: float = 0.5,
        device: str = 'cpu'
    ):
        super().__init__(
            model_paths=[model_path],
            model_class=model_class,
            temperature=temperature,
            device=device
        )


def create_ensemble_from_checkpoints(
    checkpoint_dir: str,
    pattern: str = "*.pt",
    max_models: int = 5
) -> List[str]:
    """
    Find model checkpoints for ensemble

    Args:
        checkpoint_dir: Directory containing checkpoints
        pattern: Glob pattern for checkpoint files
        max_models: Maximum number of models to include

    Returns:
        List of checkpoint paths
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_path.glob(pattern))

    # Prefer best models
    best_models = [c for c in checkpoints if 'best' in c.name]
    other_models = [c for c in checkpoints if 'best' not in c.name]

    # Prioritize best, then others
    selected = best_models[:max_models]
    if len(selected) < max_models:
        selected.extend(other_models[:max_models - len(selected)])

    return [str(p) for p in selected[:max_models]]


def main():
    """Demo temperature scaling effect"""
    print("\n" + "="*60)
    print("Temperature Scaling Demo")
    print("="*60)

    # Simulate underconfident model predictions
    raw_predictions = np.array([0.52, 0.55, 0.48, 0.58, 0.45, 0.62])

    print("\nRaw predictions (underconfident):")
    for i, p in enumerate(raw_predictions):
        print(f"  Sample {i+1}: {p:.2f}")

    print("\nAfter temperature scaling (T=0.5):")
    scaler = TemperatureScaler(temperature=0.5)
    calibrated = scaler.scale(raw_predictions)

    for i, (raw, cal) in enumerate(zip(raw_predictions, calibrated)):
        direction = "UP" if cal > 0.5 else "DOWN"
        conf = "HIGH" if abs(cal - 0.5) > 0.3 else "MEDIUM" if abs(cal - 0.5) > 0.1 else "LOW"
        print(f"  Sample {i+1}: {raw:.2f} -> {cal:.2f} ({direction}, {conf})")

    print("\n" + "-"*60)
    print("Ensemble Agreement Demo")
    print("-"*60)

    # Simulate 3 model predictions
    model1_pred = np.array([0.55, 0.52, 0.70])
    model2_pred = np.array([0.58, 0.48, 0.72])
    model3_pred = np.array([0.54, 0.55, 0.68])

    all_preds = np.vstack([model1_pred, model2_pred, model3_pred])
    mean_pred = np.mean(all_preds, axis=0)
    std_pred = np.std(all_preds, axis=0)

    print("\nModel predictions:")
    for i in range(3):
        print(f"  Sample {i+1}: M1={all_preds[0,i]:.2f}, M2={all_preds[1,i]:.2f}, M3={all_preds[2,i]:.2f}")
        print(f"           Mean={mean_pred[i]:.2f}, Std={std_pred[i]:.3f}")
        agreement = "HIGH (models agree)" if std_pred[i] < 0.03 else "LOW (models disagree)"
        print(f"           Agreement: {agreement}")
        print()


if __name__ == '__main__':
    main()
