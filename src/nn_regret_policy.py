"""Lightweight actor for NN regret checkpoints."""

from __future__ import annotations

import json
import os
import pickle
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from nn_regret import ModelConfig, Params, hash_infoset, regret_matching


class NNRegretSavedPolicy:
    """Neural regret actor usable with the Scopa web viewer."""

    def __init__(
        self,
        params: Sequence[Tuple[np.ndarray, np.ndarray]],
        avg_strategy: Optional[Dict[bytes, np.ndarray]],
        model_config: ModelConfig,
        seed: int,
    ):
        if not params:
            raise ValueError("NN regret checkpoint contains no parameters")
        self.params: Params = [
            (np.asarray(w, dtype=np.float32), np.asarray(b, dtype=np.float32))
            for w, b in params
        ]
        self.avg_strategy: Dict[bytes, np.ndarray] = {
            key: np.asarray(value, dtype=np.float32) for key, value in (avg_strategy or {}).items()
        }
        self.model_config = ModelConfig(
            obs_planes=int(model_config.obs_planes),
            obs_cards=int(model_config.obs_cards),
            action_dim=int(model_config.action_dim),
            hidden_layers=tuple(int(h) for h in model_config.hidden_layers),
            seat_embedding=bool(model_config.seat_embedding),
            mask_input=bool(model_config.mask_input),
            activation=str(model_config.activation),
        )
        # Ensure action_dim matches the parameter shapes.
        self.action_dim = int(self.params[-1][1].shape[0])
        if self.action_dim != int(self.model_config.action_dim):
            self.model_config.action_dim = self.action_dim
        self.rng = np.random.default_rng(int(seed))

    # ------------------------------------------------------------------
    def _activation(self, x: np.ndarray) -> np.ndarray:
        kind = self.model_config.activation.lower()
        if kind == "relu":
            return np.maximum(x, 0.0, out=x)
        if kind == "tanh":
            return np.tanh(x)
        if kind == "elu":
            return np.where(x > 0.0, x, np.expm1(x))
        raise ValueError(f"Unsupported activation '{self.model_config.activation}'")

    def _forward(self, obs: np.ndarray, seat: int, mask: np.ndarray) -> np.ndarray:
        parts = [obs.reshape(-1)]
        if self.model_config.seat_embedding:
            seat_vec = np.zeros(4, dtype=np.float32)
            seat_vec[int(seat) % 4] = 1.0
            parts.append(seat_vec)
        if self.model_config.mask_input:
            parts.append(mask.astype(np.float32).reshape(-1))
        x = np.concatenate(parts, axis=0).astype(np.float32)
        for idx, (w, b) in enumerate(self.params):
            x = x @ w + b
            if idx < len(self.params) - 1:
                x = self._activation(x)
        return x

    @staticmethod
    def _history_offsets() -> Dict[str, int]:
        return {
            "cuori": 0,
            "picche": 10,
            "fiori": 20,
            "bello": 30,
        }

    def _prepare_observation(self, env, seat: int, obs: np.ndarray) -> np.ndarray:
        obs_arr = np.asarray(obs, dtype=np.float32)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr.reshape(1, -1)
        planes = int(self.model_config.obs_planes)
        cards = int(self.model_config.obs_cards)
        if obs_arr.shape == (planes, cards):
            return obs_arr
        padded = np.zeros((planes, cards), dtype=np.float32)
        copy_planes = min(obs_arr.shape[0], planes)
        copy_cards = min(obs_arr.shape[1], cards)
        padded[:copy_planes, :copy_cards] = obs_arr[:copy_planes, :copy_cards]
        if env is not None and planes > copy_planes:
            players = getattr(getattr(env, "game", None), "players", None)
            if players and len(players) >= 4:
                friend = (seat + 2) % len(players)
                enemy1 = (seat + 1) % len(players)
                enemy2 = (seat + 3) % len(players)
                offset = self._history_offsets()
                if planes > 4:
                    padded[3].fill(0.0)
                    for card in players[enemy1].history:
                        idx = (card.rank - 1) + offset.get(card.suit, 0)
                        if 0 <= idx < cards:
                            padded[3][idx] = 1.0
                if planes > 5:
                    padded[4].fill(0.0)
                    for card in players[enemy2].history:
                        idx = (card.rank - 1) + offset.get(card.suit, 0)
                        if 0 <= idx < cards:
                            padded[4][idx] = 1.0
                    padded[5].fill(0.0)
                    for card in players[friend].history:
                        idx = (card.rank - 1) + offset.get(card.suit, 0)
                        if 0 <= idx < cards:
                            padded[5][idx] = 1.0
        return padded

    def _mask_from_obs(self, obs: np.ndarray) -> np.ndarray:
        first_plane = np.asarray(obs[0], dtype=np.float32).reshape(-1)
        mask = np.zeros(self.action_dim, dtype=np.float32)
        limit = min(self.action_dim, first_plane.size)
        if limit > 0:
            mask[:limit] = first_plane[:limit]
        return (mask > 0.5).astype(np.float32)

    def _choose_action(self, seat: int, obs: np.ndarray, mask: np.ndarray) -> int:
        mask = np.asarray(mask, dtype=np.float32).reshape(-1)
        if mask.size != self.action_dim:
            mask = self._mask_from_obs(obs)
        legal_idx = np.flatnonzero(mask > 0.5)
        probs = None
        if self.avg_strategy:
            key = hash_infoset(obs.astype(np.float32), int(seat))
            strat = self.avg_strategy.get(key)
            if strat is not None:
                strat = np.asarray(strat, dtype=np.float32)
                clipped = strat * mask
                total = float(clipped.sum())
                if total > 0.0:
                    probs = clipped / total
        if probs is None:
            regrets = self._forward(obs, seat, mask)
            probs = regret_matching(np.asarray(regrets, dtype=np.float32), mask)
        if legal_idx.size == 0:
            legal_idx = np.arange(self.action_dim, dtype=np.int32)
        legal_probs = probs[legal_idx]
        total = float(legal_probs.sum())
        if total <= 0.0 or not np.isfinite(total):
            return int(self.rng.choice(legal_idx))
        legal_probs = np.clip(legal_probs / total, 0.0, 1.0)
        total = float(legal_probs.sum())
        if total <= 0.0 or not np.isfinite(total):
            return int(self.rng.choice(legal_idx))
        legal_probs /= total
        return int(self.rng.choice(legal_idx, p=legal_probs))

    def act_with_env(self, env, seat: int, obs: np.ndarray, mask: np.ndarray) -> int:
        padded = self._prepare_observation(env, int(seat), obs)
        return self._choose_action(int(seat), padded, mask)

    def act_with_mask(self, seat: int, obs: np.ndarray, mask: np.ndarray) -> int:
        padded = self._prepare_observation(None, int(seat), obs)
        return self._choose_action(int(seat), padded, mask)

    def act_from_obs(self, seat: int, obs: np.ndarray) -> int:
        padded = self._prepare_observation(None, int(seat), obs)
        mask = self._mask_from_obs(padded)
        return self._choose_action(int(seat), padded, mask)


def _infer_model_config(params: Sequence[Tuple[np.ndarray, np.ndarray]]) -> ModelConfig:
    action_dim = int(params[-1][1].shape[0])
    input_dim = int(params[0][0].shape[0])
    hidden_layers = tuple(int(w.shape[1]) for w, _ in params[:-1])
    obs_cards = 40
    candidates = []
    for planes in range(4, 17):
        base = planes * obs_cards
        if base > input_dim:
            break
        remainder = input_dim - base
        for seat_flag in (True, False):
            seat_size = 4 if seat_flag else 0
            for mask_flag in (True, False):
                mask_size = action_dim if mask_flag else 0
                if base + seat_size + mask_size == input_dim:
                    candidates.append((planes, seat_flag, mask_flag))
    if candidates:
        candidates.sort(key=lambda x: (-int(x[1]), -int(x[2]), x[0]))
        planes, seat_flag, mask_flag = candidates[0]
        return ModelConfig(
            obs_planes=int(planes),
            obs_cards=obs_cards,
            action_dim=action_dim,
            hidden_layers=hidden_layers or (256, 128),
            seat_embedding=bool(seat_flag),
            mask_input=bool(mask_flag),
            activation="relu",
        )
    return ModelConfig(
        obs_planes=6,
        obs_cards=obs_cards,
        action_dim=action_dim,
        hidden_layers=hidden_layers or (256, 128),
        seat_embedding=True,
        mask_input=True,
        activation="relu",
    )


def _load_run_config(checkpoint_path: str) -> Optional[ModelConfig]:
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    run_dir = os.path.dirname(ckpt_dir)
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        model_cfg = payload.get("model")
        if not isinstance(model_cfg, dict):
            return None
        return ModelConfig(
            obs_planes=int(model_cfg.get("obs_planes", 6)),
            obs_cards=int(model_cfg.get("obs_cards", 40)),
            action_dim=int(model_cfg.get("action_dim", 40)),
            hidden_layers=tuple(int(x) for x in model_cfg.get("hidden_layers", (256, 128))),
            seat_embedding=bool(model_cfg.get("seat_embedding", True)),
            mask_input=bool(model_cfg.get("mask_input", True)),
            activation=str(model_cfg.get("activation", "relu")),
        )
    except Exception:
        return None


def load_policy(path: str, seed: int = 0) -> NNRegretSavedPolicy:
    """Load a neural regret checkpoint."""
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    if "params" not in payload:
        raise ValueError("Checkpoint missing 'params'; not an NN regret artifact")
    params = payload["params"]
    avg_strategy = payload.get("avg_strategy") or {}
    model_config = _load_run_config(path)
    if model_config is None:
        model_config = _infer_model_config(params)
    return NNRegretSavedPolicy(params=params, avg_strategy=avg_strategy, model_config=model_config, seed=seed)
