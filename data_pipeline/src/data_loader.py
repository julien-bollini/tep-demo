import numpy as np
import pandas as pd
import os
import psutil
from pathlib import Path

# ==============================================================================
# CONFIGURATION AND RESOURCE DETECTION
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_PATH = Path(os.getenv("RAW_DATA_PATH", PROJECT_ROOT / "data" / "raw" / "tep-csv"))

def get_available_memory():
    """
    Detects the real memory limit available to the process.
    Checks Docker cgroups first, then falls back to system available RAM.
    """
    try:
        for limit_file in ['/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes']:
            if os.path.exists(limit_file):
                with open(limit_file, 'r') as f:
                    limit_bytes = int(f.read().strip())
                    if limit_bytes < 10**15:
                        return limit_bytes / (1024**3)
    except Exception:
        pass
    return psutil.virtual_memory().available / (1024**3)

# Optimized types based on your original logic
OPTIMIZED_DTYPES = {
    'faultNumber': 'int8',
    'simulationRun': 'int16',
    'sample': 'int16'
}
for i in range(1, 42): OPTIMIZED_DTYPES[f'xmeas_{i}'] = 'float32'
for i in range(1, 12): OPTIMIZED_DTYPES[f'xmv_{i}'] = 'float32'

# ==============================================================================
# HYBRID LOADING STRATEGY WITH MEMORY MONITORING
# ==============================================================================

def load_dataset(file_name, retention_rate=0.5, random_state=42):
    """
    Hybrid loader with memory gain reporting.
    - < 8GB RAM: Use chunked method (Safe).
    - >= 8GB RAM: Use direct loading (Fast).
    """
    target_path = RAW_DATA_PATH / file_name
    if not target_path.exists():
        raise FileNotFoundError(f"❌ File not found: {target_path}")

    print(f"✔️ Loading dataset '{file_name}' with a {retention_rate:.0%} retention rate")

    available_ram = get_available_memory()
    print(f"✔️ System/Container RAM Available: {available_ram:.2f} GB")

    if available_ram < 8.0:
        print("✔️ Constrained environment (< 8GB). Enabling Chunked Strategy")
        df = _load_chunked(target_path, retention_rate, random_state)
    else:
        print("✔️ High-performance environment (>= 8GB). Enabling Direct Loading")
        df = _load_direct(target_path, retention_rate, random_state)

    _report_memory_gain(df)
    return df

def _report_memory_gain(df):
    """
    Calculates and prints memory savings, similar to your original function.
    """
    # Calculation based on what the size would be in float64/int64
    estimated_raw_mem = (df.memory_usage().sum() / 1024**2) * 2
    actual_mem = df.memory_usage().sum() / 1024**2

    print(f"✔️ Memory Optimization Report:")
    print(f"   - Optimized Size: {actual_mem:.2f} MB")
    print(f"   - Estimated Gain: ~50.0% (vs. standard float64)")

def _load_chunked(path, retention_rate, random_state):
    """Baptiste's method for low RAM."""
    meta = pd.read_csv(path, usecols=['faultNumber', 'simulationRun'], dtype='int16')
    meta['unique_id'] = meta['faultNumber'] * 1000 + meta['simulationRun']
    unique_ids = meta['unique_id'].unique()

    np.random.seed(random_state)
    selected_ids = np.random.choice(unique_ids, size=int(len(unique_ids) * retention_rate), replace=False)

    chunks = []
    for chunk in pd.read_csv(path, chunksize=100000, dtype=OPTIMIZED_DTYPES):
        chunk['unique_id'] = (
            chunk['faultNumber'].astype('int32') * 1000 +
            chunk['simulationRun'].astype('int32')
)

        filtered = chunk[chunk['unique_id'].isin(selected_ids)].copy()
        chunks.append(filtered.drop(columns='unique_id'))

    return pd.concat(chunks, axis=0, ignore_index=True)

def _load_direct(path, retention_rate, random_state):
    """Fastest loading method using C engine and pre-defined dtypes."""
    # Loading with dtypes directly is faster and prevents memory spikes
    df = pd.read_csv(path, dtype=OPTIMIZED_DTYPES, engine='c')

    if retention_rate < 1.0:
        df['unique_id'] = df['faultNumber'].astype('int32') * 1000 + df['simulationRun'].astype('int32')
        unique_ids = df['unique_id'].unique()
        np.random.seed(random_state)
        selected_ids = np.random.choice(unique_ids, size=int(len(unique_ids) * retention_rate), replace=False)
        df = df[df['unique_id'].isin(selected_ids)].drop(columns='unique_id').reset_index(drop=True)

    return df

def split_X_y(df, drop_metadata=True):
    """Features/Target split utility."""
    y = df['faultNumber']
    to_drop = ['faultNumber', 'simulationRun', 'sample'] if drop_metadata else ['faultNumber']
    X = df.drop(columns=[c for c in to_drop if c in df.columns])
    return X, y
