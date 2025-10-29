"""Train the neural regret approximation pipeline for Scopa."""

import argparse
import dataclasses
import json
import os
import time
from typing import Optional

import numpy as np

from tlogger import TLogger
import env as env_mod

from nn_regret import CheckpointManager, ModelConfig, NNRegretTrainer, OptimizerConfig, TrainerConfig


def _params_to_numpy(params):
    return [(np.asarray(w), np.asarray(b)) for (w, b) in params]


def card_str_list(cards):
    return [str(c) for c in cards]


def action_to_card(player, action_idx: int):
    for c in player.hand:
        ind = (c.rank - 1) + {
            "cuori": 0,
            "picche": 10,
            "fiori": 20,
            "bello": 30,
        }[c.suit]
        if ind == action_idx:
            return c
    return None


def _pad_observation(env, seat: int, obs: np.ndarray, planes: int, cards: int) -> np.ndarray:
    """Extend an environment observation to the model observation shape."""

    model_obs = np.zeros((planes, cards), dtype=np.float32)
    limit = min(obs.shape[0], planes)
    model_obs[:limit] = obs[:limit]
    if planes > limit:
        offset = {"cuori": 0, "picche": 10, "fiori": 20, "bello": 30}
        players = env.game.players
        friend = (seat + 2) % 4
        enemy1 = (seat + 1) % 4
        enemy2 = (seat + 3) % 4
        if planes > 4:
            for card in players[enemy1].history:
                idx = (card.rank - 1) + offset[card.suit]
                model_obs[3][idx] = 1.0
        if planes > 5:
            for card in players[enemy2].history:
                idx = (card.rank - 1) + offset[card.suit]
                model_obs[4][idx] = 1.0
        if planes > 5:
            for card in players[friend].history:
                idx = (card.rank - 1) + offset[card.suit]
                model_obs[5][idx] = 1.0
    return model_obs


def preview_game(env, trainer: NNRegretTrainer, label: str = "", episodes: int = 1, dump_dir: Optional[str] = None):
    for epi in range(episodes):
        env.reset()
        transcript = []
        steps = 0
        while True:
            agent = env.agent_selection
            if env.terminations[agent] or env.truncations[agent]:
                break
            seat = env.agent_name_mapping[agent]
            obs = env.observations[agent]
            mask = env.infos[agent]["action_mask"]
            model_obs = _pad_observation(env, seat, obs, trainer.model.config.obs_planes, trainer.model.config.obs_cards)
            action = trainer.act_from_obs(seat, model_obs, mask, explore=False)
            legal = [i for i, m in enumerate(mask) if m == 1]
            action_source = "policy"
            fallback_used = False
            if mask[action] == 0:
                fallback_used = True
                action = legal[0] if legal else int(action)
            card = action_to_card(env.game.players[seat], action)
            capture_desc = ""
            if card is not None:
                if card.rank == 1:
                    capture_desc = "(Ace: captures all table)"
                else:
                    isin, comb = env.game.card_in_table(card)
                    if isin:
                        capture_desc = f"(captures {[str(c) for c in comb]})"
            step_record = {
                "step": steps,
                "agent": agent,
                "seat": seat,
                "hand": card_str_list(env.game.players[seat].hand),
                "table_before": card_str_list(env.game.table),
                "legal_actions": legal,
                "action_index": int(action),
                "action_source": action_source,
                "fallback_used": fallback_used,
                "action_card": str(card) if card is not None else None,
                "capture_description": capture_desc,
                "action_mask": list(mask),
            }
            env.step(action)
            step_record["table_after"] = card_str_list(env.game.table)
            transcript.append(step_record)
            steps += 1
        result_scores = list(env.roundScores())
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
            payload = {
                "episode": epi,
                "label": label,
                "steps": transcript,
                "result_scores": result_scores,
            }
            filename = f"{label or 'policy'}_preview_{epi:03d}.json"
            with open(os.path.join(dump_dir, filename), "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)



def parse_layers(arg: str):
    stripped = arg.strip()
    if not stripped:
        return ()
    return tuple(int(x) for x in stripped.split(",") if x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=600, help="Training iterations")
    parser.add_argument("--deals_per_iter", type=int, default=12, help="Deals (trajectories) per iteration")
    parser.add_argument("--updates_per_iter", type=int, default=6, help="Gradient steps per iteration")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--buffer_capacity", type=int, default=8192)
    parser.add_argument("--mc_rollouts", type=int, default=4, help="Monte Carlo rollouts for bootstrap targets")
    parser.add_argument("--exploration", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.2)
    parser.add_argument("--layers", type=str, default="512,256,128", help="Comma separated hidden sizes")
    parser.add_argument("--target_update", type=int, default=50, help="Frequency of target param sync (0 disables)")
    parser.add_argument("--bootstrap_mc", action="store_true", default=True, help="Enable Monte Carlo bootstrap for regret targets")
    parser.add_argument("--seed", type=int, default=time.time_ns())
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--eval_eps", type=int, default=128, help="Episodes for evaluation sweeps")
    parser.add_argument("--preview_policy", type=int, default=0, help="Number of preview games with learned policy")
    parser.add_argument("--preview_dump_dir", type=str, default="", help="Directory to dump preview transcripts")
    parser.add_argument("--save_dir", type=str, default="", help="Directory to save checkpoints and configs")
    parser.add_argument("--run_name", type=str, default="nn_regret", help="Run directory name")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_root = os.path.join(repo_root, "runs", args.run_name)
    os.makedirs(run_root, exist_ok=True)
    run_dir = os.path.join(run_root, time.strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)

    tlog = TLogger(log_dir=run_dir)

    hidden_layers = parse_layers(args.layers)
    model_config = ModelConfig(hidden_layers=hidden_layers)
    opt_config = OptimizerConfig(
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    trainer_config = TrainerConfig(
        seed=args.seed,
        deals_per_iter=args.deals_per_iter,
        mc_rollouts=max(1, args.mc_rollouts),
        exploration=args.exploration,
        batch_size=args.batch_size,
        updates_per_iter=args.updates_per_iter,
        buffer_capacity=args.buffer_capacity,
        target_update_every=max(0, args.target_update),
        bootstrap_monte_carlo=args.bootstrap_mc,
    )

    trainer = NNRegretTrainer(model_config, opt_config, trainer_config, tlogger=tlog)

    checkpoint_dir = args.save_dir if args.save_dir else os.path.join(run_dir, "checkpoints")
    ckpt = CheckpointManager(checkpoint_dir)

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "model": dataclasses.asdict(model_config),
                "optimizer": dataclasses.asdict(opt_config),
                "trainer": dataclasses.asdict(trainer_config),
            },
            fh,
            indent=2,
        )

    trainer.train(iterations=args.iters, log_every=args.log_every)

    avg_strategy = trainer.snapshot_average_strategy()
    ckpt.save(
        "final.pkl",
        {
            "params": _params_to_numpy(trainer.params),
            "avg_strategy": avg_strategy,
        },
    )

    if args.preview_policy > 0:
        env = env_mod.env(tlogger=tlog)
        preview_dump = args.preview_dump_dir if args.preview_dump_dir else os.path.join(run_dir, "previews")
        preview_game(env, trainer, label="policy", episodes=args.preview_policy, dump_dir=preview_dump)

    print(f"Run artifacts stored in {run_dir}")


if __name__ == "__main__":
    main()
