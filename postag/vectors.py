"""Sparse feature encoding and persisted, reusable dimensionality reduction."""

import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.utils.sparsefuncs import mean_variance_axis


VARIANTS = ("token", "context", "context_tf", "context_tf_identity")


def select_features(features, variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown feature variant: {variant}")
    return {key: value for key, value in features.items()
            if key != "bias"
            and (variant != "token" or not key.startswith(("-1:", "+1:")))
            and (variant in ("context_tf", "context_tf_identity")
                 or not key.endswith("word.tf"))
            and (variant == "context_tf_identity" or not key.endswith("word.lower()"))}


class FeatureEncoder(BaseEstimator, TransformerMixin):
    """DictVectorizer with optional std scaling of ONLY length and TF columns.

    Scaling does not center data or change categorical/Boolean features. No
    sparse feature matrix is converted to dense before SVD.
    """

    def __init__(self, variant="context_tf", numeric_scaling="none"):
        self.variant = variant
        self.numeric_scaling = numeric_scaling

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        if self.numeric_scaling not in ("none", "std"):
            raise ValueError("numeric_scaling must be none or std")
        self.vectorizer_ = DictVectorizer(dtype=np.float64, sparse=True, sort=True)
        matrix = self.vectorizer_.fit_transform(select_features(f, self.variant) for f in X)
        self.scales_ = np.ones(matrix.shape[1])
        self.numeric_ranges_ = {}
        _, variances = mean_variance_axis(matrix, axis=0)
        for i, name in enumerate(self.vectorizer_.get_feature_names_out()):
            if name.endswith(("word.length", "word.tf")):
                column = matrix[:, i]
                self.numeric_ranges_[name] = {"min": float(column.min()),
                                              "max": float(column.max()),
                                              "std": float(np.sqrt(variances[i]))}
                if self.numeric_scaling == "std" and variances[i] > 1e-20:
                    self.scales_[i] = np.sqrt(variances[i])
        return matrix.multiply(1 / self.scales_).tocsr()

    def transform(self, X):
        matrix = self.vectorizer_.transform(select_features(f, self.variant) for f in X)
        return matrix.multiply(1 / self.scales_).tocsr()


def fit_reducer(matrix, requested_components, seed=42, normalize=False):
    if requested_components < 1:
        raise ValueError("SVD component count must be positive")
    if min(matrix.shape) < 2:
        raise ValueError("SVD requires at least two rows and two feature columns")
    # Leave one dimension spare and record the actual dimension on small inputs.
    components = min(requested_components, min(matrix.shape) - 1)
    reducer = Pipeline([
        ("svd", TruncatedSVD(n_components=components, random_state=seed)),
        ("normalize", Normalizer() if normalize else "passthrough"),
    ])
    # Some macOS BLAS builds emit matmul warnings even for finite results. Keep
    # every warning in the report and validate the actual fitted outputs rather
    # than globally disabling NumPy's numerical warnings.
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        reduced = reducer.fit_transform(matrix)
    svd = reducer.named_steps["svd"]
    if not all(np.isfinite(value).all() for value in
               (reduced, svd.components_, svd.singular_values_)):
        raise ValueError("SVD produced non-finite vectors or fitted parameters")
    gram = np.einsum("ik,jk->ij", svd.components_, svd.components_, optimize=False)
    orthogonality_error = float(np.max(np.abs(gram - np.eye(len(gram)))))
    if orthogonality_error > 1e-6:
        raise ValueError(f"SVD components failed orthogonality check: {orthogonality_error}")
    variance = svd.explained_variance_ratio_
    # A corpus of identical feature rows has zero variance, so the ratio is undefined.
    explained = float(variance.sum()) if np.isfinite(variance).all() else None
    return reducer, reduced, {"requested_components": requested_components,
                              "components": reduced.shape[1],
                              "explained_variance_ratio": explained,
                              "warnings": sorted({str(w.message) for w in captured}),
                              "finite_vectors_and_parameters": True,
                              "component_orthogonality_max_error": orthogonality_error,
                              "normalization": "l2" if normalize else "none"}
