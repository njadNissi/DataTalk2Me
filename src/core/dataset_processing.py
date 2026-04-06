import pandas as pd

def get_representative_sample(df, target_col=None, sample_size=150):
    """
    Returns a FAST, representative sample that INCLUDES ALL UNIQUE VALUES
    No lag, shows full dataset diversity
    """
    if target_col is None or target_col not in df.columns:
        # If no target, return random sample
        return df.sample(min(sample_size, len(df)), random_state=42)

    # Get ALL unique classes
    unique_vals = df[target_col].unique()
    n_unique = len(unique_vals)

    # Take a few rows from EACH CLASS
    samples = []
    for val in unique_vals:
        class_subset = df[df[target_col] == val]
        # Take at least 2 rows per class, adjust total
        take_n = max(2, sample_size // (n_unique * 2))
        samples.append(class_subset.sample(min(take_n, len(class_subset)), random_state=42))

    # Combine + shuffle
    sample_df = pd.concat(samples).sample(frac=1, random_state=42)
    return sample_df
