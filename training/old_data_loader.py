def split_X_y(df, drop_metadata=True):
    """Features/Target split utility."""
    y = df['faultNumber']
    to_drop = ['faultNumber', 'simulationRun', 'sample'] if drop_metadata else ['faultNumber']
    X = df.drop(columns=[c for c in to_drop if c in df.columns])
    return X, y
