import numpy as np
from typing import Tuple
from dataclasses import dataclass

@dataclass
class NNIndex:
    model: object
    metric: str
    use_faiss: bool
    dim: int

def build_index(X: np.ndarray, use_faiss: bool = True, metric: str = "cosine") -> NNIndex:
    try:
        if use_faiss:
            import faiss  # type: ignore
            dim = X.shape[1]
            idx = faiss.IndexFlatIP(dim)
            Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
            idx.add(Xn.astype('float32'))
            return NNIndex(model=idx, metric=metric, use_faiss=True, dim=dim)
    except Exception:
        pass
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=50, metric=metric).fit(X)
    return NNIndex(model=nn, metric=metric, use_faiss=False, dim=X.shape[1])

def query_index(index: NNIndex, Xq: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    if index.use_faiss:
        Xq = Xq / (np.linalg.norm(Xq, axis=1, keepdims=True) + 1e-8)
        D, I = index.model.search(Xq.astype('float32'), k)
        D = 1 - D
        return D, I
    else:
        D, I = index.model.kneighbors(Xq, n_neighbors=k, return_distance=True)
        return D, I
