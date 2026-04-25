from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from vantage6.algorithm.tools.decorators import data
from vantage6.algorithm.tools.util import error, info, warn


@data(1)
def partial(
    df1: pd.DataFrame,
    label_col: str,
    features: Optional[List[str]] = None,
) -> Any:
    """
    Compute local sufficient statistics for federated LDA.

    Returns per-class:
    - sample count
    - sum vector
    - within-class scatter matrix
    """
    info("Starting partial LDA statistics computation")

    if df1 is None or df1.empty:
        warn("Received empty dataframe.")
        return {"error": "Empty dataframe."}

    if label_col not in df1.columns:
        error(f"label_col '{label_col}' not found in dataframe")
        return {"error": f"label_col '{label_col}' not found in dataframe"}

    if features is not None:
        missing = [c for c in features if c not in df1.columns]
        if missing:
            error(f"Requested features not found in data: {missing}")
            return {"error": f"Requested features not found: {missing}"}
        X = df1[features]
        columns = list(features)
    else:
        X = df1.select_dtypes(include=[np.number])
        columns = [c for c in X.columns if c != label_col]
        X = X[columns]

    if X.empty:
        error("No usable numeric features found for LDA.")
        return {"error": "No usable numeric features found for LDA."}

    local = pd.concat([df1[[label_col]], X], axis=1).dropna(axis=0, how="any")
    if local.empty:
        error("All rows contain NaNs for selected features/label.")
        return {"error": "All rows contain NaNs for selected features/label."}

    class_stats: Dict[str, Dict[str, Any]] = {}
    d = len(columns)

    for cls, grp in local.groupby(label_col):
        x = grp[columns].to_numpy(dtype=float)
        n = int(x.shape[0])
        mean = np.mean(x, axis=0)
        centered = x - mean
        sw = centered.T @ centered

        class_stats[str(cls)] = {
            "n": n,
            "sum": np.sum(x, axis=0).tolist(),
            "sw": sw.tolist(),
        }

    return {
        "columns": columns,
        "classes": sorted(class_stats.keys()),
        "class_stats": class_stats,
    }
