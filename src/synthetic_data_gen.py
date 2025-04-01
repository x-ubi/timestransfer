import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess
from typing import Tuple, List


class SyntheticTimeSeriesGenerator:
    def __init__(self, seed=None):
        """Initialize the generator with optional random seed"""
        self.rng = np.random.RandomState(seed)
        self.max_context_length = 2048
        self.armas = {}
        self.trends = {}
        self.sines = {}
        self.cosines = {}

    def generate_series(self, n_series: int = 1) -> List[np.ndarray]:
        """Generate multiple synthetic time series"""
        return [self._generate_single_series(i) for i in range(n_series)]

    def _generate_single_series(self, ctr) -> np.ndarray:
        """Generate a single synthetic time series of length 2048"""
        components = []
        weights = []

        # Randomly enable/disable each component (I-IV)
        enable_trend = self.rng.choice([True, False])
        enable_arma = self.rng.choice([True, False])
        enable_sine = self.rng.choice([True, False])
        enable_cosine = self.rng.choice([True, False])

        # Generate each enabled component
        if enable_trend:
            trend = self._generate_trend_component()
            components.append(trend)
            weights.append(self.rng.uniform(0.1, 1.0))
            self.trends[ctr] = trend

            # 50% chance to apply trend multiplicatively
            if self.rng.choice([True, False]):
                self.multiplicative_trend = True
            else:
                self.multiplicative_trend = False

        if enable_arma:
            arma = self._generate_arma_component()
            components.append(arma)
            self.armas[ctr] = arma
            weights.append(self.rng.uniform(0.1, 1.0))

        if enable_sine:
            sine = self._generate_seasonal_component(wave_type="sine")
            components.append(sine)
            self.sines[ctr] = sine
            weights.append(self.rng.uniform(0.1, 1.0))

        if enable_cosine:
            cosine = self._generate_seasonal_component(wave_type="cosine")
            components.append(cosine)
            self.cosines[ctr] = cosine
            weights.append(self.rng.uniform(0.1, 1.0))

        # If no components were enabled, default to ARMA
        if not components:
            arma = self._generate_arma_component()
            components.append(arma)
            self.armas[ctr] = arma
            weights.append(1.0)

        # Combine components with weights
        combined = np.zeros(self.max_context_length)
        for comp, weight in zip(components, weights):
            combined += comp * weight

        # Apply multiplicative trend if specified
        if enable_trend and self.multiplicative_trend and len(components) > 1:
            normalized_trend = trend - np.min(trend) + 0.1  # ensure positive
            combined = (combined - trend * weights[0]) * (1 + normalized_trend * weights[0])

        return combined

    def _generate_trend_component(self) -> np.ndarray:
        """Generate piece-wise linear trend component (I)"""
        n_segments = self.rng.randint(2, 9)  # between 2 and 8
        breakpoints = sorted(
            self.rng.choice(np.arange(1, self.max_context_length - 1), size=n_segments - 1, replace=False)
        )
        breakpoints = [np.int64(0)] + breakpoints + [np.int64(self.max_context_length)]

        trend = np.zeros(self.max_context_length)
        current_value = self.rng.normal(0, 1)

        for i in range(len(breakpoints) - 1):
            start_idx, end_idx = breakpoints[i], breakpoints[i + 1]
            segment_length = end_idx - start_idx

            slope = 0
            while np.abs(slope) < 0.05:
                slope = self.rng.normal(0, 0.1)

            # Generate linear segment
            segment = current_value + slope * np.arange(segment_length)
            trend[start_idx:end_idx] = segment

            # Update current value for next segment
            current_value = segment[-1]

        return trend

    def _generate_arma_component(self) -> np.ndarray:
        """Generate ARMA(p,q) component (II)"""
        p = self.rng.randint(1, 9)  # 1 ≤ p ≤ 8
        q = self.rng.randint(1, 9)  # 1 ≤ q ≤ 8

        # Generate coefficients from normal or uniform distribution
        if self.rng.choice([True, False]):
            ar_params = self.rng.normal(0, 0.2, p)
            ma_params = self.rng.normal(0, 0.2, q)
        else:
            ar_params = self.rng.uniform(-0.5, 0.5, p)
            ma_params = self.rng.uniform(-0.5, 0.5, q)

        # Normalize coefficients to ensure stationarity/invertibility
        ar_params = ar_params / (1 + np.sum(np.abs(ar_params)))
        ma_params = ma_params / (1 + np.sum(np.abs(ma_params)))

        # Generate ARMA process
        ar = np.r_[1, -ar_params]  # AR parameters
        ma = np.r_[1, ma_params]  # MA parameters
        arma_process = ArmaProcess(ar, ma)

        # Generate the time series
        arma_series = arma_process.generate_sample(nsample=self.max_context_length)

        return arma_series

    def _generate_seasonal_component(self, wave_type: str) -> np.ndarray:
        """Generate seasonal component - sine (III) or cosine (IV) wave"""
        period = self.rng.randint(4, self.max_context_length // 2)
        time_delay = self.rng.randint(0, period)
        amplitude = self.rng.uniform(0.5, 2.0)
        phase = self.rng.uniform(0, 2 * np.pi)

        t = np.arange(self.max_context_length) + time_delay

        if wave_type == "sine":
            wave = amplitude * np.sin(2 * np.pi * t / period + phase)
        elif wave_type == "cosine":
            wave = amplitude * np.cos(2 * np.pi * t / period + phase)
        else:
            raise ValueError("wave_type must be 'sine' or 'cosine'")

        return wave
