import random
import time
import statistics

# ─────────────────────────────────────────────
# STAGE 1 — Generate synthetic CPU metric data
# ─────────────────────────────────────────────

def generate_cpu_metrics(num_points: int = 100) -> list[float]:
    """
    Generates a synthetic CPU usage time series.
    Normal range: 40–65% with small random variation.
    Anomalies are injected at fixed positions by the caller — 
    this function produces only the clean baseline.
    """
    metrics = []
    for _ in range(num_points):
        # Normal CPU usage — baseline 50%, ±15% random variation
        value = 50.0 + random.uniform(-15, 15)
        metrics.append(round(value, 2))
    return metrics


def inject_anomalies(metrics: list[float]) -> list[float]:
    """
    Injects three types of anomalies into the metric series:
    - Point anomaly at position 20: single spike to 95%
    - Point anomaly at position 50: single drop to 5%
    - Collective anomaly at positions 70–75: steady upward drift
    """
    metrics[20] = 95.0   # sudden spike — possible runaway process
    metrics[50] = 5.0    # sudden drop — possible service crash
    
    # Collective anomaly — gradual memory leak pattern
    for i in range(70, 76):
        metrics[i] = 65.0 + (i - 70) * 5.0  # 65, 70, 75, 80, 85, 90
    
    return metrics

# ─────────────────────────────────────────────
# STAGE 2 — Detection functions
# ─────────────────────────────────────────────

def detect_zscore(metrics: list[float], threshold: float = 2.5) -> list[int]:
    """
    Z-score detection — flags any point more than `threshold`
    standard deviations away from the mean of the entire series.
    Returns a list of anomalous indices.
    """
    mean = statistics.mean(metrics)
    stdev = statistics.stdev(metrics)
    
    anomalies = []
    for i, value in enumerate(metrics):
        # Z-score = how many standard deviations from the mean
        z_score = abs(value - mean) / stdev
        if z_score > threshold:
            anomalies.append(i)
    
    return anomalies


def detect_rolling_average(
    metrics: list[float],
    window: int = 10,
    threshold: float = 20.0
) -> list[int]:
    """
    Rolling average detection — compares each point against
    the average of the previous `window` points.
    Flags points that deviate by more than `threshold` percent.
    Only starts after enough points exist to fill the window.
    """
    anomalies = []
    
    for i in range(window, len(metrics)):
        # Calculate average of the previous `window` points
        window_values = metrics[i - window:i]
        rolling_mean = statistics.mean(window_values)
        
        # How far is the current value from the rolling mean?
        deviation = abs(metrics[i] - rolling_mean)
        
        # Express deviation as percentage of rolling mean
        deviation_pct = (deviation / rolling_mean) * 100
        
        if deviation_pct > threshold:
            anomalies.append(i)
    
    return anomalies

# ─────────────────────────────────────────────
# STAGE 3 — Main execution
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Log Intelligence Assistant — Anomaly Detector")
    print("=" * 55)
    print("Generating synthetic CPU metrics (100 time steps)...")
    time.sleep(0.5)

    # Generate clean baseline then inject known anomalies
    metrics = generate_cpu_metrics(100)
    metrics = inject_anomalies(metrics)

    print("Anomalies injected at positions 20, 50, and 70–75.")
    print("Running detection...\n")
    time.sleep(0.5)

    # Run both detectors
    zscore_anomalies = detect_zscore(metrics, threshold=2.5)
    rolling_anomalies = detect_rolling_average(metrics, window=10, threshold=20.0)

    # Combine both result sets for display
    all_flagged = set(zscore_anomalies) | set(rolling_anomalies)

    # Print each time step — flag anomalies as they appear
    print(f"{'Step':<6} {'CPU %':<10} {'Z-Score':<10} {'Rolling':<10} {'Status'}")
    print("-" * 55)

    for i, value in enumerate(metrics):
        z_flag = "ANOMALY" if i in zscore_anomalies else "normal"
        r_flag = "ANOMALY" if i in rolling_anomalies else "normal"

        # Only print anomalous rows and every 10th normal row
        # This keeps terminal output readable
        if i in all_flagged or i % 10 == 0:
            status = "⚠ FLAGGED" if i in all_flagged else ""
            print(f"{i:<6} {value:<10} {z_flag:<10} {r_flag:<10} {status}")
            time.sleep(0.05)  # slight delay — simulates real-time stream output

    print("-" * 55)
    print(f"\nZ-Score detector flagged:        {len(zscore_anomalies)} anomalies at positions {zscore_anomalies}")
    print(f"Rolling average detector flagged: {len(rolling_anomalies)} anomalies at positions {rolling_anomalies}")
    print(f"Combined unique anomalies:        {len(all_flagged)} at positions {sorted(all_flagged)}")
    print("\nDetection complete.")


if __name__ == "__main__":
    main()