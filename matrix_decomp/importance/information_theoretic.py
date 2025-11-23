"""
Information-theoretic importance scoring methods.

These methods use information theory concepts to measure importance.

Methods implemented:
- Mutual Information: I(Layer; Output)
- Entropy-based Importance
- Information Bottleneck Principle
- Representation Similarity Analysis
"""

from typing import Dict, Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from scipy.stats import entropy


class MutualInformationImportance:
    """
    Mutual Information between layer activations and outputs.

    Measures how much information about the output is contained in each layer.

    I(X;Y) = H(Y) - H(Y|X)

    Higher mutual information = more important layer.
    """

    def compute_scores(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: str = "cpu",
        num_bins: int = 20,
        max_batches: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Compute mutual information scores.

        Args:
            model: Model to analyze
            data_loader: DataLoader
            device: Device
            num_bins: Number of bins for histogram estimation
            max_batches: Max batches
            verbose: Show progress

        Returns:
            Dictionary mapping layer names to MI scores
        """
        model = model.to(device)
        model.eval()

        # Collect activations
        activations = {}
        outputs_all = []

        def get_activation_hook(name):
            def hook(module, input, output):
                # Store flattened activations
                act = output.detach().cpu().flatten().numpy()
                if name not in activations:
                    activations[name] = []
                activations[name].append(act)
            return hook

        # Register hooks
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                handle = module.register_forward_hook(get_activation_hook(name))
                handles.append(handle)

        # Collect data
        iterator = enumerate(data_loader)
        if verbose:
            iterator = tqdm(iterator, desc="Collecting activations", total=len(data_loader))

        with torch.no_grad():
            for batch_idx, batch in iterator:
                if max_batches is not None and batch_idx >= max_batches:
                    break

                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                else:
                    inputs = batch

                inputs = inputs.to(device)
                outputs = model(inputs)
                outputs_all.append(outputs.cpu().flatten().numpy())

        # Remove hooks
        for handle in handles:
            handle.remove()

        # Concatenate all outputs
        outputs_all = np.concatenate(outputs_all)

        # Compute mutual information for each layer
        mi_scores = {}

        for name, act_list in activations.items():
            if not act_list:
                continue

            # Concatenate activations
            act_concat = np.concatenate(act_list)

            # Subsample if too large
            if len(act_concat) > 100000:
                indices = np.random.choice(len(act_concat), 100000, replace=False)
                act_concat = act_concat[indices]
                outputs_sample = outputs_all[:len(indices)]
            else:
                outputs_sample = outputs_all[:len(act_concat)]

            # Estimate MI using binned histogram
            mi = self._estimate_mi(act_concat, outputs_sample, num_bins)
            mi_scores[name] = mi

        # Normalize
        max_score = max(mi_scores.values()) if mi_scores else 1.0
        if max_score > 0:
            mi_scores = {k: v / max_score for k, v in mi_scores.items()}

        return mi_scores

    def _estimate_mi(self, x: np.ndarray, y: np.ndarray, num_bins: int) -> float:
        """
        Estimate mutual information using histogram-based method.

        I(X;Y) = H(X) + H(Y) - H(X,Y)
        """
        # Discretize into bins
        x_binned = np.digitize(x, bins=np.linspace(x.min(), x.max(), num_bins))
        y_binned = np.digitize(y, bins=np.linspace(y.min(), y.max(), num_bins))

        # Compute entropies
        h_x = entropy(np.bincount(x_binned) + 1e-10)
        h_y = entropy(np.bincount(y_binned) + 1e-10)

        # Joint entropy
        xy_joint = x_binned * num_bins + y_binned
        h_xy = entropy(np.bincount(xy_joint) + 1e-10)

        # Mutual information
        mi = h_x + h_y - h_xy

        return max(0, mi)  # MI is non-negative


class EntropyBasedImportance:
    """
    Entropy of layer activations.

    High entropy layers capture more diverse information.

    H(X) = -Σ p(x) log p(x)
    """

    def compute_scores(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: str = "cpu",
        num_bins: int = 50,
        max_batches: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Compute entropy-based importance.

        Args:
            model: Model to analyze
            data_loader: DataLoader
            device: Device
            num_bins: Number of bins for histogram
            max_batches: Max batches
            verbose: Show progress

        Returns:
            Dictionary mapping layer names to entropy scores
        """
        model = model.to(device)
        model.eval()

        # Collect activations
        activations = {}

        def get_activation_hook(name):
            def hook(module, input, output):
                act = output.detach().cpu().flatten().numpy()
                if name not in activations:
                    activations[name] = []
                activations[name].append(act)
            return hook

        # Register hooks
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                handle = module.register_forward_hook(get_activation_hook(name))
                handles.append(handle)

        # Collect data
        iterator = enumerate(data_loader)
        if verbose:
            iterator = tqdm(iterator, desc="Computing entropy", total=len(data_loader))

        with torch.no_grad():
            for batch_idx, batch in iterator:
                if max_batches is not None and batch_idx >= max_batches:
                    break

                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                else:
                    inputs = batch

                inputs = inputs.to(device)
                _ = model(inputs)

        # Remove hooks
        for handle in handles:
            handle.remove()

        # Compute entropy for each layer
        entropy_scores = {}

        for name, act_list in activations.items():
            if not act_list:
                continue

            # Concatenate activations
            act_concat = np.concatenate(act_list)

            # Subsample if needed
            if len(act_concat) > 100000:
                act_concat = np.random.choice(act_concat, 100000, replace=False)

            # Compute histogram
            hist, _ = np.histogram(act_concat, bins=num_bins)
            prob = hist / hist.sum()

            # Compute entropy
            ent = entropy(prob + 1e-10)
            entropy_scores[name] = ent

        # Normalize
        max_score = max(entropy_scores.values()) if entropy_scores else 1.0
        if max_score > 0:
            entropy_scores = {k: v / max_score for k, v in entropy_scores.items()}

        return entropy_scores


class RepresentationSimilarityImportance:
    """
    Representation Similarity Analysis (RSA).

    Measures how similar layer representations are to the final output.

    Uses correlation between representational distance matrices.
    """

    def compute_scores(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: str = "cpu",
        max_samples: int = 1000,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Compute RSA-based importance.

        Args:
            model: Model to analyze
            data_loader: DataLoader
            device: Device
            max_samples: Max samples for RSA (RSA is O(n²))
            verbose: Show progress

        Returns:
            Dictionary mapping layer names to RSA similarity scores
        """
        model = model.to(device)
        model.eval()

        # Collect representations
        representations = {}
        final_outputs = []

        def get_activation_hook(name):
            def hook(module, input, output):
                # Flatten spatial dimensions but keep batch
                act = output.detach().cpu()
                if act.dim() > 2:
                    act = act.flatten(1)  # [batch, features]
                if name not in representations:
                    representations[name] = []
                representations[name].append(act)
            return hook

        # Register hooks
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                handle = module.register_forward_hook(get_activation_hook(name))
                handles.append(handle)

        # Collect data
        num_collected = 0

        with torch.no_grad():
            for batch in data_loader:
                if num_collected >= max_samples:
                    break

                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                else:
                    inputs = batch

                inputs = inputs.to(device)
                outputs = model(inputs)

                final_outputs.append(outputs.cpu().flatten(1))
                num_collected += inputs.size(0)

        # Remove hooks
        for handle in handles:
            handle.remove()

        # Concatenate final outputs
        final_outputs = torch.cat(final_outputs, dim=0)[:max_samples]

        # Compute RDM for final output
        final_rdm = self._compute_rdm(final_outputs)

        # Compute RSA for each layer
        rsa_scores = {}

        if verbose:
            print("Computing RSA scores...")

        for name, repr_list in representations.items():
            if not repr_list:
                continue

            # Concatenate representations
            repr_concat = torch.cat(repr_list, dim=0)[:max_samples]

            # Compute RDM for this layer
            layer_rdm = self._compute_rdm(repr_concat)

            # Compute correlation between RDMs
            similarity = self._rdm_correlation(layer_rdm, final_rdm)
            rsa_scores[name] = similarity

        # Normalize
        max_score = max(rsa_scores.values()) if rsa_scores else 1.0
        if max_score > 0:
            rsa_scores = {k: v / max_score for k, v in rsa_scores.items()}

        return rsa_scores

    def _compute_rdm(self, representations: torch.Tensor) -> np.ndarray:
        """
        Compute Representational Dissimilarity Matrix.

        RDM[i,j] = distance(repr[i], repr[j])
        """
        # Compute pairwise distances
        n = representations.size(0)
        rdm = torch.zeros(n, n)

        for i in range(n):
            for j in range(i+1, n):
                dist = torch.norm(representations[i] - representations[j])
                rdm[i, j] = dist
                rdm[j, i] = dist

        return rdm.numpy()

    def _rdm_correlation(self, rdm1: np.ndarray, rdm2: np.ndarray) -> float:
        """Compute correlation between two RDMs."""
        # Flatten upper triangle
        mask = np.triu_indices_from(rdm1, k=1)
        vec1 = rdm1[mask]
        vec2 = rdm2[mask]

        # Compute correlation
        if len(vec1) > 1:
            corr = np.corrcoef(vec1, vec2)[0, 1]
            return abs(corr)  # Use absolute correlation
        return 0.0


class ActivationSparsityImportance:
    """
    APoZ: Average Percentage of Zeros.

    Measures activation sparsity. Less sparse = more important.

    Paper: "Network Trimming: A Data-Driven Neuron Pruning Approach"
    """

    def compute_scores(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: str = "cpu",
        max_batches: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Compute activation sparsity (APoZ) scores.

        Args:
            model: Model to analyze
            data_loader: DataLoader
            device: Device
            max_batches: Max batches
            verbose: Show progress

        Returns:
            Dictionary mapping layer names to sparsity scores
            (Note: Lower sparsity = higher importance, so we return 1 - sparsity)
        """
        model = model.to(device)
        model.eval()

        # Track zero activations
        zero_counts = {}
        total_counts = {}

        def get_activation_hook(name):
            def hook(module, input, output):
                act = output.detach()
                zeros = (act == 0).sum().item()
                total = act.numel()

                if name not in zero_counts:
                    zero_counts[name] = 0
                    total_counts[name] = 0

                zero_counts[name] += zeros
                total_counts[name] += total

            return hook

        # Register hooks
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.ReLU)):
                handle = module.register_forward_hook(get_activation_hook(name))
                handles.append(handle)

        # Collect data
        iterator = enumerate(data_loader)
        if verbose:
            iterator = tqdm(iterator, desc="Computing APoZ", total=len(data_loader))

        with torch.no_grad():
            for batch_idx, batch in iterator:
                if max_batches is not None and batch_idx >= max_batches:
                    break

                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                else:
                    inputs = batch

                inputs = inputs.to(device)
                _ = model(inputs)

        # Remove hooks
        for handle in handles:
            handle.remove()

        # Compute sparsity scores
        sparsity_scores = {}

        for name in zero_counts:
            if total_counts[name] > 0:
                apoz = zero_counts[name] / total_counts[name]
                # Convert to importance: less sparse = more important
                importance = 1.0 - apoz
                sparsity_scores[name] = importance

        # Normalize
        max_score = max(sparsity_scores.values()) if sparsity_scores else 1.0
        if max_score > 0:
            sparsity_scores = {k: v / max_score for k, v in sparsity_scores.items()}

        return sparsity_scores
