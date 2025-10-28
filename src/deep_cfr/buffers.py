"""
Replay buffers for Deep CFR: RegretMemory and PolicyMemory.
"""

import random
from typing import List, Optional

import numpy as np
import torch


class ReservoirBuffer:
    """
    Classic reservoir sampling buffer with fixed capacity.
    Keeps an unbiased sample of a stream of items of unknown length.

    :param capacity: maximum number of items to store
    :param seed: optional random seed for reproducibility
    """

    def __init__(self, capacity: int, seed: Optional[int] = None):
        self.capacity = capacity
        self.storage: List = []
        self.n_seen = 0
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.storage)

    def add(self, item):
        """
        Add an item to the reservoir buffer.
        :param item: item to add
        """
        self.n_seen += 1
        if len(self.storage) < self.capacity:
            self.storage.append(item)
        else:
            j = self.rng.randint(1, self.n_seen)
            if j <= self.capacity:
                self.storage[j - 1] = item

    def sample(self, batch_size: int):
        """
        Sample a batch of items from the buffer.

        :param batch_size: number of items to sample
        :return: list of sampled items
        """
        batch = self.rng.sample(self.storage, k=min(batch_size, len(self.storage)))
        return batch


class RegretMemory:
    """
    Stores tuples: (info_tensor, legal_mask, advantages)
      - info_tensor: torch.float (in_dim,)
      - legal_mask : torch.float (A,)
      - advantages : torch.float (A,)  # instantaneous regrets A_p

    :param capacity: maximum number of items to store
    :param device: torch device for sampling
    :param seed: optional random seed for reproducibility
    """

    def __init__(self, capacity: int, device: str = "cpu", seed: Optional[int] = None):
        self.buf = ReservoirBuffer(capacity, seed)
        self.device = device

    def add(self, info_np: np.ndarray, mask_np: np.ndarray, adv_np: np.ndarray):
        self.buf.add(
            (
                torch.from_numpy(info_np).float(),
                torch.from_numpy(mask_np).float(),
                torch.from_numpy(adv_np).float(),
            )
        )

    def sample(self, batch_size: int):
        batch = self.buf.sample(batch_size)
        infos, masks, advs = zip(*batch)
        return (
            torch.stack(infos).to(self.device),
            torch.stack(masks).to(self.device),
            torch.stack(advs).to(self.device),
        )


class PolicyMemory:
    """
    Stores tuples: (info_tensor, legal_mask, target_probs, weight)
      - target_probs: torch.float (A,)  # σ target at this infoset
      - weight      : float             # reach weight (for weighted CE)
    
    :param capacity: maximum number of items to store
    :param device: torch device for sampling
    :param seed: optional random seed for reproducibility
    """

    def __init__(self, capacity: int, device: str = "cpu", seed: Optional[int] = None):
        self.buf = ReservoirBuffer(capacity, seed)
        self.device = device

    def add(
        self,
        info_np: np.ndarray,
        mask_np: np.ndarray,
        probs_np: np.ndarray,
        weight: float,
    ):
        """
        Add a policy sample to the memory.

        :param info_np: observation array
        :param mask_np: legal action mask array
        :param probs_np: target policy probabilities array
        :param weight: reach weight
        """
        self.buf.add(
            (
                torch.from_numpy(info_np).float(),
                torch.from_numpy(mask_np).float(),
                torch.from_numpy(probs_np).float(),
                float(weight),
            )
        )

    def sample(self, batch_size: int):
        """
        Sample a batch of policy samples from the memory.

        :param batch_size: number of samples to draw
        :return: tuple of tensors (infos, masks, probs, weights)
        """
        batch = self.buf.sample(batch_size)
        infos, masks, probs, weights = zip(*batch)
        return (
            torch.stack(infos).to(self.device),
            torch.stack(masks).to(self.device),
            torch.stack(probs).to(self.device),
            torch.tensor(weights, dtype=torch.float32, device=self.device),
        )
