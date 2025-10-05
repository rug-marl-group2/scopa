import argparse
import time
import random
from typing import Optional

import numpy as np

from tlogger import TLogger
from env import env as make_env
from cfr_jax import CFRTrainer, SavedPolicy


def card_str_list(cards):
    return [str(c) for c in cards]


def action_to_card(player, action_idx: int):
    for c in player.hand:
        ind = (c.rank - 1) + {
            'cuori': 0,
            'picche': 10,
            'fiori': 20,
            'bello': 30
        }[c.suit]
        if ind == action_idx:
            return c
    return None


def select_action_for_seat(mode: str, seat: int, obs: np.ndarray, mask: np.ndarray,
                           actor: Optional[object]) -> int:
    legal = [i for i, m in enumerate(mask) if m == 1]
    if mode == "vs_random" and (seat % 2 == 1):
        return random.choice(legal)
    # default to policy actor
    if actor is None:
        return random.choice(legal)
    a = actor.act_from_obs(seat, obs)
    if mask[a] == 0:
        # Safety fallback
        a = legal[0]
    return a


def play_match_once(env, actor: object, mode: str = "selfplay", verbose: bool = False):
    steps = 0
    if verbose:
        print(f"\n=== Play match ({mode}) ===")
    env.reset()
    while True:
        agent = env.agent_selection
        if env.terminations[agent] or env.truncations[agent]:
            break
        seat = env.agent_name_mapping[agent]
        player = env.game.players[seat]
        obs = env.observations[agent]
        mask = env.infos[agent]["action_mask"]

        action = select_action_for_seat(mode, seat, obs, mask, actor)

        if verbose:
            card = action_to_card(player, action)
            capture_desc = ""
            if card is not None:
                if card.rank == 1:
                    capture_desc = "(Ace: captures all table)"
                else:
                    isin, comb = env.game.card_in_table(card)
                    if isin:
                        capture_desc = f"(captures {[str(c) for c in comb]})"
            print(f"Step {steps:02d} | {agent} plays {card} {capture_desc}")
            print(f"  Hand: {card_str_list(player.hand)}")
            print(f"  Table before: {card_str_list(env.game.table)}")

        env.step(action)

        if verbose:
            print(f"  Table after:  {card_str_list(env.game.table)}")
        steps += 1

    scores = env.roundScores()
    print("Result scores:", scores)
    return scores


def main():
    parser = argparse.ArgumentParser(description="Play Scopa matches from a saved CFR checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved checkpoint (.pkl)")
    parser.add_argument("--mode", type=str, default="selfplay", choices=["selfplay", "vs_random"], help="Self-play or play seats 0/2 vs random 1/3")
    parser.add_argument("--games", type=int, default=1, help="Number of matches to play")
    parser.add_argument("--seed", type=int, default=123, help="Seed for randomness")
    parser.add_argument("--verbose", action="store_true", help="Print step-by-step moves")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Logger + env
    tlog = TLogger(log_dir="runs/infer/" + time.strftime("%Y-%m-%d-%H-%M-%S"))
    env = make_env(tlog)

    # Load policy (works for both avg and full checkpoints)
    actor: SavedPolicy = CFRTrainer.load_avg_policy(args.checkpoint, seed=args.seed)

    # Play N games
    for gi in range(args.games):
        if args.verbose and args.games > 1:
            print(f"\n=== Game {gi+1}/{args.games} ===")
        play_match_once(env, actor, mode=args.mode, verbose=args.verbose)

    tlog.close()


if __name__ == "__main__":
    main()

