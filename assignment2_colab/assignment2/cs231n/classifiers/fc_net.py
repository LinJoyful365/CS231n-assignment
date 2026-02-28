from builtins import range
from builtins import object
import os
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design.

    Architecture: affine - relu - affine - softmax
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        self.params = {}
        self.reg = reg

        self.params["W1"] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params["b1"] = np.zeros(hidden_dim)
        self.params["W2"] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params["b2"] = np.zeros(num_classes)

    def loss(self, X, y=None):
        scores = None

        hidden_out, hidden_cache = affine_relu_forward(
            X, self.params["W1"], self.params["b1"]
        )
        scores, score_cache = affine_forward(
            hidden_out, self.params["W2"], self.params["b2"]
        )

        if y is None:
            return scores

        loss, grads = 0.0, {}
        data_loss, dscore = softmax_loss(scores, y)
        reg_loss = 0.5 * self.reg * (
            np.sum(self.params["W1"] * self.params["W1"])
            + np.sum(self.params["W2"] * self.params["W2"])
        )
        loss = data_loss + reg_loss

        dhidden, dW2, db2 = affine_backward(dscore, score_cache)
        _, dW1, db1 = affine_relu_backward(dhidden, hidden_cache)

        grads["W2"] = dW2 + self.reg * self.params["W2"]
        grads["b2"] = db2
        grads["W1"] = dW1 + self.reg * self.params["W1"]
        grads["b1"] = db1

        return loss, grads

    def save(self, fname):
        """Save model parameters."""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        params = self.params
        np.save(fpath, params)
        print(fname, "saved.")

    def load(self, fname):
        """Load model parameters."""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        if not os.path.exists(fpath):
            print(fname, "not available.")
            return False
        else:
            params = np.load(fpath, allow_pickle=True).item()
            self.params = params
            print(fname, "loaded.")
            return True


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    For a network with L layers, the architecture is

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        layer_dims = [input_dim] + list(hidden_dims) + [num_classes]
        for layer_idx in range(1, self.num_layers + 1):
            in_dim = layer_dims[layer_idx - 1]
            out_dim = layer_dims[layer_idx]
            self.params[f"W{layer_idx}"] = weight_scale * np.random.randn(in_dim, out_dim)
            self.params[f"b{layer_idx}"] = np.zeros(out_dim)

            if self.normalization in ("batchnorm", "layernorm") and layer_idx < self.num_layers:
                self.params[f"gamma{layer_idx}"] = np.ones(out_dim)
                self.params[f"beta{layer_idx}"] = np.zeros(out_dim)

        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for _ in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for _ in range(self.num_layers - 1)]

        for param_key, param_val in self.params.items():
            self.params[param_key] = param_val.astype(dtype)

    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode

        scores = None

        layer_input = X
        hidden_cache_dict = {}
        dropout_cache_dict = {}

        for layer_idx in range(1, self.num_layers):
            weight_key = f"W{layer_idx}"
            bias_key = f"b{layer_idx}"

            affine_out, affine_cache = affine_forward(
                layer_input, self.params[weight_key], self.params[bias_key]
            )

            norm_cache = None
            if self.normalization == "batchnorm":
                gamma_key = f"gamma{layer_idx}"
                beta_key = f"beta{layer_idx}"
                affine_out, norm_cache = batchnorm_forward(
                    affine_out,
                    self.params[gamma_key],
                    self.params[beta_key],
                    self.bn_params[layer_idx - 1],
                )
            elif self.normalization == "layernorm":
                gamma_key = f"gamma{layer_idx}"
                beta_key = f"beta{layer_idx}"
                affine_out, norm_cache = layernorm_forward(
                    affine_out,
                    self.params[gamma_key],
                    self.params[beta_key],
                    self.bn_params[layer_idx - 1],
                )

            relu_out, relu_cache = relu_forward(affine_out)

            if self.use_dropout:
                # 这里缓存拆开存，回传的时候可读性好一点
                relu_out, do_cache = dropout_forward(relu_out, self.dropout_param)
                dropout_cache_dict[layer_idx] = do_cache

            hidden_cache_dict[layer_idx] = (affine_cache, norm_cache, relu_cache)
            layer_input = relu_out

        final_weight_key = f"W{self.num_layers}"
        final_bias_key = f"b{self.num_layers}"
        scores, final_cache = affine_forward(
            layer_input, self.params[final_weight_key], self.params[final_bias_key]
        )

        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        data_loss, upstream_grad = softmax_loss(scores, y)

        reg_loss = 0.0
        for layer_idx in range(1, self.num_layers + 1):
            weight_key = f"W{layer_idx}"
            reg_loss += 0.5 * self.reg * np.sum(
                self.params[weight_key] * self.params[weight_key]
            )
        loss = data_loss + reg_loss

        dx, dW, db = affine_backward(upstream_grad, final_cache)
        grads[final_weight_key] = dW + self.reg * self.params[final_weight_key]
        grads[final_bias_key] = db

        for layer_idx in range(self.num_layers - 1, 0, -1):
            if self.use_dropout:
                dx = dropout_backward(dx, dropout_cache_dict[layer_idx])

            affine_cache, norm_cache, relu_cache = hidden_cache_dict[layer_idx]
            dx = relu_backward(dx, relu_cache)

            if self.normalization == "batchnorm":
                dx, dgamma, dbeta = batchnorm_backward_alt(dx, norm_cache)
                grads[f"gamma{layer_idx}"] = dgamma
                grads[f"beta{layer_idx}"] = dbeta
            elif self.normalization == "layernorm":
                dx, dgamma, dbeta = layernorm_backward(dx, norm_cache)
                grads[f"gamma{layer_idx}"] = dgamma
                grads[f"beta{layer_idx}"] = dbeta

            dx, dW, db = affine_backward(dx, affine_cache)
            grads[f"W{layer_idx}"] = dW + self.reg * self.params[f"W{layer_idx}"]
            grads[f"b{layer_idx}"] = db

        return loss, grads

    def save(self, fname):
        """Save model parameters."""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        params = self.params
        np.save(fpath, params)
        print(fname, "saved.")

    def load(self, fname):
        """Load model parameters."""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        if not os.path.exists(fpath):
            print(fname, "not available.")
            return False
        else:
            params = np.load(fpath, allow_pickle=True).item()
            self.params = params
            print(fname, "loaded.")
            return True
