"""Small regressors (numpy only) and their ONNX export.

Every exported graph maps `features` float32[N, FEATURE_COUNT] to `adjustments` float32[N, 10]
in slider units; the runtime only clamps and rounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .schema import FEATURE_COUNT, OUTPUT_KEYS

OPSET = 17
IR_VERSION = 8  # loadable by onnxruntime >= 1.14, including the 1.22 runtime used by `ort`
INPUT_NAME = "features"
OUTPUT_NAME = "adjustments"


@dataclass
class Scaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Scaler":
        scale = values.std(axis=0)
        return cls(values.mean(axis=0), np.where(scale < 1e-6, 1.0, scale))

    def apply(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale


class Model:
    kind = "abstract"

    def fit(self, features: np.ndarray, targets: np.ndarray, seed: int) -> "Model":
        raise NotImplementedError

    def predict(self, features: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params(self) -> dict:
        return {}

    def to_onnx(self) -> onnx.ModelProto:
        raise NotImplementedError


class MeanBaseline(Model):
    kind = "mean"

    def fit(self, features, targets, seed=0):
        self.mean = targets.mean(axis=0)
        return self

    def predict(self, features):
        return np.tile(self.mean, (features.shape[0], 1))


# --- ONNX helpers ---------------------------------------------------------------------------


@dataclass
class _Graph:
    nodes: list = field(default_factory=list)
    initializers: list = field(default_factory=list)
    counter: int = 0

    def const(self, name: str, value: np.ndarray, dtype=np.float32) -> str:
        self.initializers.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name))
        return name

    def op(self, op_type: str, inputs: list[str], output: str | None = None, **attributes) -> str:
        self.counter += 1
        output = output or f"{op_type.lower()}_{self.counter}"
        self.nodes.append(helper.make_node(op_type, inputs, [output], **attributes))
        return output

    def standardize(self, scaler: Scaler) -> str:
        centered = self.op("Sub", [INPUT_NAME, self.const("feature_mean", scaler.mean)])
        return self.op("Div", [centered, self.const("feature_scale", scaler.scale)])

    def finish(self, name: str, last: str, doc: str) -> onnx.ModelProto:
        assert self.nodes[-1].output[0] == last, "the output must be produced by the last node"
        self.nodes[-1].output[0] = OUTPUT_NAME
        graph = helper.make_graph(
            self.nodes,
            name,
            [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, ["N", FEATURE_COUNT])],
            [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, ["N", len(OUTPUT_KEYS)])],
            self.initializers,
        )
        model = helper.make_model(
            graph,
            producer_name="picportal-style-trainer",
            opset_imports=[helper.make_opsetid("", OPSET)],
            doc_string=doc,
        )
        model.ir_version = IR_VERSION
        onnx.checker.check_model(model)
        return model

    def unstandardize(self, values: str, scaler: Scaler) -> str:
        scaled = self.op("Mul", [values, self.const("target_scale", scaler.scale)])
        return self.op("Add", [scaled, self.const("target_mean", scaler.mean)])


# --- Models ---------------------------------------------------------------------------------


class Ridge(Model):
    kind = "ridge"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, features, targets, seed=0):
        self.x_scaler, self.y_scaler = Scaler.fit(features), Scaler.fit(targets)
        z, y = self.x_scaler.apply(features), self.y_scaler.apply(targets)
        gram = z.T @ z + self.alpha * np.eye(z.shape[1])
        self.weights = np.linalg.solve(gram, z.T @ y)
        return self

    def predict(self, features):
        z = self.x_scaler.apply(features)
        return self.y_scaler.mean + (z @ self.weights) * self.y_scaler.scale

    def params(self):
        return {"alpha": self.alpha}

    def to_onnx(self):
        g = _Graph()
        z = g.standardize(self.x_scaler)
        y = g.op("MatMul", [z, g.const("weights", self.weights)])
        return g.finish("style_ridge", g.unstandardize(y, self.y_scaler), "ridge regression")


class Knn(Model):
    """Mean of the k nearest training images in standardized feature space."""

    kind = "knn"

    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, features, targets, seed=0):
        self.x_scaler = Scaler.fit(features)
        self.references = self.x_scaler.apply(features)
        self.values = targets.copy()
        self.k = min(self.k, len(features))
        return self

    def _neighbours(self, features: np.ndarray) -> np.ndarray:
        z = self.x_scaler.apply(features)
        distances = (z**2).sum(axis=1, keepdims=True) - 2 * z @ self.references.T + (self.references**2).sum(axis=1)
        return np.argsort(distances, axis=1, kind="stable")[:, : self.k]

    def predict(self, features):
        return self.values[self._neighbours(features)].mean(axis=1)

    def params(self):
        return {"k": self.k}

    def to_onnx(self, name: str = "style_knn"):
        g = _Graph()
        z = g.standardize(self.x_scaler)
        squared = g.op("Mul", [z, z])
        norms = g.op("ReduceSum", [squared, g.const("axis_1", [1], np.int64)], keepdims=1)
        cross = g.op("MatMul", [z, g.const("references_t", self.references.T)])
        cross2 = g.op("Mul", [cross, g.const("minus_two", -2.0)])
        partial = g.op("Add", [norms, cross2])
        distances = g.op("Add", [partial, g.const("reference_norms", (self.references**2).sum(axis=1))])
        negated = g.op("Neg", [distances])
        g.nodes.append(
            helper.make_node(
                "TopK",
                [negated, g.const("k", [self.k], np.int64)],
                ["topk_values", "topk_indices"],
                axis=1,
                largest=1,
                sorted=1,
            )
        )
        gathered = g.op("Gather", [g.const("reference_targets", self.values), "topk_indices"], axis=0)
        mean = g.op("ReduceMean", [gathered], axes=[1], keepdims=0)
        return g.finish(name, mean, f"k-nearest neighbours (k={self.k})")


class ClusterMean(Knn):
    """Fallback for small corpora: k-means in feature space, predict the cluster's mean edit."""

    kind = "cluster_mean"

    def __init__(self, clusters: int | None = None):
        super().__init__(k=1)
        self.clusters = clusters

    def fit(self, features, targets, seed=0):
        count = len(features)
        clusters = self.clusters or int(np.clip(round(count / 10), 1, 5))
        clusters = min(clusters, count)
        scaler = Scaler.fit(features)
        z = scaler.apply(features)
        rng = np.random.default_rng(seed)
        centroids = z[[rng.integers(count)]]
        while len(centroids) < clusters:  # k-means++ seeding
            nearest = ((z[:, None, :] - centroids[None]) ** 2).sum(axis=2).min(axis=1)
            probabilities = nearest / nearest.sum() if nearest.sum() > 0 else None
            centroids = np.vstack([centroids, z[rng.choice(count, p=probabilities)]])
        for _ in range(50):
            assignment = ((z[:, None, :] - centroids[None]) ** 2).sum(axis=2).argmin(axis=1)
            updated = np.array(
                [z[assignment == c].mean(axis=0) if np.any(assignment == c) else centroids[c] for c in range(clusters)]
            )
            if np.allclose(updated, centroids):
                break
            centroids = updated
        assignment = ((z[:, None, :] - centroids[None]) ** 2).sum(axis=2).argmin(axis=1)
        means = np.array(
            [targets[assignment == c].mean(axis=0) if np.any(assignment == c) else targets.mean(axis=0) for c in range(clusters)]
        )
        self.x_scaler, self.references, self.values, self.k = scaler, centroids, means, 1
        self.sizes = [int(np.sum(assignment == c)) for c in range(clusters)]
        return self

    def params(self):
        return {"clusters": int(len(self.references)), "clusterSizes": self.sizes}

    def to_onnx(self, name: str = "style_cluster_mean"):
        return super().to_onnx(name)


class Mlp(Model):
    """One hidden tanh layer trained with full-batch Adam on standardized targets."""

    kind = "mlp"

    def __init__(self, hidden: int = 32, weight_decay: float = 1e-3, epochs: int = 800, learning_rate: float = 1e-2):
        self.hidden, self.weight_decay, self.epochs, self.learning_rate = hidden, weight_decay, epochs, learning_rate

    def fit(self, features, targets, seed=0):
        self.x_scaler, self.y_scaler = Scaler.fit(features), Scaler.fit(targets)
        z, y = self.x_scaler.apply(features), self.y_scaler.apply(targets)
        rng = np.random.default_rng(seed)
        inputs, outputs = z.shape[1], y.shape[1]
        params = [
            rng.normal(0, 1 / np.sqrt(inputs), (inputs, self.hidden)),
            np.zeros(self.hidden),
            rng.normal(0, 1 / np.sqrt(self.hidden), (self.hidden, outputs)),
            np.zeros(outputs),
        ]
        first, second = [np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params]
        beta1, beta2, count = 0.9, 0.999, len(z)
        for step in range(1, self.epochs + 1):
            w1, b1, w2, b2 = params
            hidden = np.tanh(z @ w1 + b1)
            error = (hidden @ w2 + b2 - y) * (2.0 / (count * outputs))
            grad_w2 = hidden.T @ error + self.weight_decay * w2
            grad_hidden = (error @ w2.T) * (1 - hidden**2)
            grad_w1 = z.T @ grad_hidden + self.weight_decay * w1
            grads = [grad_w1, grad_hidden.sum(axis=0), grad_w2, error.sum(axis=0)]
            for index, grad in enumerate(grads):
                first[index] = beta1 * first[index] + (1 - beta1) * grad
                second[index] = beta2 * second[index] + (1 - beta2) * grad**2
                corrected = first[index] / (1 - beta1**step)
                params[index] -= self.learning_rate * corrected / (np.sqrt(second[index] / (1 - beta2**step)) + 1e-8)
        self.params_ = params
        return self

    def predict(self, features):
        w1, b1, w2, b2 = self.params_
        z = self.x_scaler.apply(features)
        return self.y_scaler.mean + (np.tanh(z @ w1 + b1) @ w2 + b2) * self.y_scaler.scale

    def params(self):
        return {"hidden": self.hidden, "weightDecay": self.weight_decay, "epochs": self.epochs}

    def to_onnx(self):
        w1, b1, w2, b2 = self.params_
        g = _Graph()
        z = g.standardize(self.x_scaler)
        hidden = g.op("Tanh", [g.op("Add", [g.op("MatMul", [z, g.const("w1", w1)]), g.const("b1", b1)])])
        y = g.op("Add", [g.op("MatMul", [hidden, g.const("w2", w2)]), g.const("b2", b2)])
        return g.finish("style_mlp", g.unstandardize(y, self.y_scaler), f"MLP {self.hidden} tanh")


def candidates(sample_count: int) -> list[Model]:
    models: list[Model] = [Ridge(alpha) for alpha in (0.1, 1.0, 10.0, 100.0)]
    models += [Knn(k) for k in (3, 5, 8, 12) if k < sample_count]
    models += [Mlp(hidden, decay) for hidden in (16, 32) for decay in (1e-3, 1e-2)]
    return models
