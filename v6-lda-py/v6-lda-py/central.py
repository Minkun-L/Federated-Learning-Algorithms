from typing import Any, Dict, List, Optional

import numpy as np
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.tools.exceptions import UserInputError
from vantage6.algorithm.tools.util import error, info


@algorithm_client
def central(
    client: AlgorithmClient,
    label_col: str,
    features: Optional[List[str]] = None,
    n_components: Optional[int] = None,
    reg: float = 1e-6,
) -> Any:
    """
    Central federated LDA.

    Aggregates per-node class statistics and solves
    eig(inv(SW + reg*I) @ SB).
    """
    info("Starting central federated LDA")

    organizations = client.organization.list()
    org_ids = [organization.get("id") for organization in organizations]
    if not org_ids:
        return {"error": "No organizations found in the collaboration."}

    task = client.task.create(
        input_={
            "method": "partial",
            "kwargs": {
                "label_col": label_col,
                "features": features,
            },
        },
        organizations=org_ids,
        name="Federated LDA partial stats",
        description="Compute local class sufficient statistics for federated LDA",
    )
    results = client.wait_for_results(task_id=task.get("id"))
    if not results:
        return {"error": "No results received from partial tasks."}

    columns = None
    classes = set()

    global_n: Dict[str, int] = {}
    global_sum: Dict[str, np.ndarray] = {}
    global_sw: Dict[str, np.ndarray] = {}

    for idx, res in enumerate(results):
        if not res or "error" in res:
            continue

        cols = res.get("columns")
        stats = res.get("class_stats")
        if cols is None or stats is None:
            continue

        if columns is None:
            columns = cols
        elif cols != columns:
            error(f"Column mismatch for node result index={idx}")
            return {"error": "Column mismatch between nodes."}

        d = len(columns)
        for cls, s in stats.items():
            n = int(s["n"])
            sum_vec = np.asarray(s["sum"], dtype=float)
            sw = np.asarray(s["sw"], dtype=float)

            if sum_vec.shape != (d,) or sw.shape != (d, d):
                raise UserInputError("Shape mismatch in partial statistics.")

            classes.add(cls)
            global_n[cls] = global_n.get(cls, 0) + n
            global_sum[cls] = global_sum.get(cls, np.zeros(d, dtype=float)) + sum_vec
            global_sw[cls] = global_sw.get(cls, np.zeros((d, d), dtype=float)) + sw

    if columns is None or len(classes) < 2:
        return {"error": "Need at least 2 valid classes across nodes."}

    d = len(columns)
    class_list = sorted(classes)

    means: Dict[str, np.ndarray] = {}
    total_n = 0
    total_sum = np.zeros(d, dtype=float)

    for cls in class_list:
        n = global_n[cls]
        if n <= 0:
            continue
        means[cls] = global_sum[cls] / n
        total_n += n
        total_sum += global_sum[cls]

    if total_n <= 1 or len(means) < 2:
        return {"error": "Insufficient samples after aggregation."}

    mu = total_sum / total_n

    sw = np.zeros((d, d), dtype=float)
    for cls in class_list:
        if cls in global_sw:
            sw += global_sw[cls]

    sb = np.zeros((d, d), dtype=float)
    for cls in class_list:
        n = global_n[cls]
        if n <= 0 or cls not in means:
            continue
        dm = (means[cls] - mu).reshape(-1, 1)
        sb += n * (dm @ dm.T)

    sw_reg = sw + float(reg) * np.eye(d)
    mat = np.linalg.pinv(sw_reg) @ sb

    eigvals, eigvecs = np.linalg.eig(mat)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    max_k = min(len(class_list) - 1, d)
    if n_components is None:
        k = max_k
    else:
        k = max(1, min(int(n_components), max_k))

    scalings = eigvecs[:, :k]
    scores = np.maximum(eigvals[:k], 0.0)
    denom = np.sum(np.maximum(eigvals, 0.0))
    ratio = (scores / denom) if denom > 0 else np.zeros_like(scores)

    info("Central federated LDA finished")
    return {
        "columns": columns,
        "classes": class_list,
        "n_total": int(total_n),
        "class_counts": {cls: int(global_n[cls]) for cls in class_list},
        "overall_mean": mu.tolist(),
        "class_means": {cls: means[cls].tolist() for cls in class_list},
        "scalings": scalings.tolist(),
        "explained_discriminability": scores.tolist(),
        "explained_discriminability_ratio": ratio.tolist(),
        "regularization": float(reg),
    }
