import argparse
import time
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import numpy as np

try:
    from flask import Flask, request, jsonify, Response
except Exception as e:  # pragma: no cover
    raise SystemExit("Flask is required. Install with: pip install flask")

from tlogger import TLogger
from env import env as make_env
from cfr_jax import CFRTrainer
from ctde_trainer import CTDETrainer


def load_any_policy(checkpoint: str, seed: int):
    """Attempt to load a policy checkpoint with CFR fallback before CTDE."""
    load_attempts = [
        ("CFR average policy", lambda: CFRTrainer.load_avg_policy(checkpoint, seed=seed)),
        ("CTDE actor", lambda: CTDETrainer.load_policy(checkpoint, seed=seed)),
    ]
    errors = []
    for label, loader in load_attempts:
        try:
            actor = loader()
            print(f"Loaded {label} from {checkpoint}")
            return actor
        except Exception as exc:
            errors.append(f"{label}: {exc}")
    raise RuntimeError("; ".join(errors))


def card_id(card) -> str:
    return f"{card.suit}-{int(card.rank)}"


def card_to_index(card) -> int:
    return (card.rank - 1) + {
        'cuori': 0,
        'picche': 10,
        'fiori': 20,
        'bello': 30
    }[card.suit]


def serialize_cards(cards) -> List[Dict[str, Any]]:
    out = []
    for c in cards:
        out.append({
            'id': card_id(c),
            'rank': int(c.rank),
            'suit': c.suit,
            'text': str(c),
        })
    return out


@dataclass
class GameManager:
    tlog: TLogger
    env: Any
    mode: str  # 'selfplay' or 'vs_random'
    seed: int
    actors: Dict[int, Optional[Any]] = field(default_factory=lambda: {0: None, 1: None})
    checkpoints: Dict[int, Optional[str]] = field(default_factory=lambda: {0: None, 1: None})
    last_move: Optional[Dict[str, Any]] = None

    def reset(self, mode: Optional[str] = None, seed: Optional[int] = None) -> None:
        if mode is not None:
            self.mode = mode
        if seed is not None:
            self.seed = int(seed)
        # Advance seed for variety
        self.seed += 1
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.env.reset(seed=self.seed)
        self.last_move = None

    def load_checkpoint(self, team: int, checkpoint: Optional[str]) -> None:
        if team not in self.actors:
            raise ValueError("team must be 0 or 1")
        if checkpoint:
            actor = load_any_policy(checkpoint, seed=int(self.seed) + int(team) * 7)
            self.actors[team] = actor
            self.checkpoints[team] = checkpoint
        else:
            self.actors[team] = None
            self.checkpoints[team] = None

    def _select_action(self, seat: int, obs: np.ndarray, mask: np.ndarray) -> int:
        legal = [i for i, m in enumerate(mask) if m == 1]
        team = seat % 2
        actor = self.actors.get(team)
        if self.mode == 'vs_random' and team == 1 and actor is None:
            return int(random.choice(legal))
        if actor is None:
            return int(random.choice(legal))
        if hasattr(actor, 'act_with_mask'):
            try:
                a = int(actor.act_with_mask(seat, obs, mask))
            except TypeError:
                a = int(actor.act_from_obs(seat, obs))
        else:
            a = int(actor.act_from_obs(seat, obs))
        if mask[a] == 0:
            return int(legal[0])
        return a

    def step(self) -> None:
        agent = self.env.agent_selection
        if self.env.terminations[agent] or self.env.truncations[agent]:
            return
        seat = self.env.agent_name_mapping[agent]
        player = self.env.game.players[seat]
        obs = self.env.observations[agent]
        mask = self.env.infos[agent]["action_mask"]

        action = self._select_action(seat, obs, mask)
        # Determine capture set for highlight
        highlight_ids = []
        played_id = None
        for c in list(player.hand):
            if card_to_index(c) == action:
                played_id = card_id(c)
                if c.rank == 1:
                    highlight_ids = [card_id(tc) for tc in self.env.game.table]
                else:
                    isin, comb = self.env.game.card_in_table(c)
                    if isin:
                        highlight_ids = [card_id(x) for x in comb]
                break

        self.env.step(action)

        self.last_move = {
            'agent': agent,
            'seat': int(seat),
            'played': played_id,
            'captured': highlight_ids,
        }

    def state(self) -> Dict[str, Any]:
        agent = self.env.agent_selection
        done = False
        if self.env.terminations[agent] or self.env.truncations[agent]:
            done = True
        players = self.env.game.players
        return {
            'agent': agent,
            'done': done,
            'mode': self.mode,
            'seed': int(self.seed),
            'table': serialize_cards(self.env.game.table),
            'players': [
                {
                    'name': p.name,
                    'side': int(p.side),
                    'hand': serialize_cards(p.hand),
                } for p in players
            ],
            'last_move': self.last_move,
            'scores': self.env.roundScores() if done else None,
            'checkpoints': {
                0: self.checkpoints.get(0),
                1: self.checkpoints.get(1),
            },
        }


HTML_PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Scopa Web Viewer</title>
  <style>
    body { font-family: system-ui, -apple-system, Arial, sans-serif; margin: 0; background: #ecf0f1; }
    .toolbar { display: flex; gap: 8px; align-items: center; padding: 10px; background: #fff; border-bottom: 1px solid #ccc; }
    .status { margin-left: auto; color: #2c3e50; }
    .board { display: grid; grid-template-rows: 1fr auto 1fr; grid-template-columns: 1fr 1fr 1fr; height: calc(100vh - 54px); }
    .zone { padding: 8px; border: 2px solid transparent; border-radius: 8px; transition: border-color 0.2s ease, background-color 0.2s ease; }
    .zone.active-player { border-color: #2980b9; background: rgba(41, 128, 185, 0.12); }
    .cards-row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    .col { display: flex; flex-direction: column; gap: 8px; align-items: center; }
    .card { width: 64px; height: 92px; border: 2px solid #2c3e50; border-radius: 6px; background: #fff; display: flex; align-items: center; justify-content: center; text-align: center; font-weight: bold; }
    .inner { width: 58px; height: 86px; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: #000; }
    .hl { border-color: #e67e22; }
    .s-cuori { background: #e74c3c; }
    .s-picche { background: #2c3e50; color: #ecf0f1; }
    .s-fiori { background: #27ae60; }
    .s-bello { background: #f1c40f; }
    .title { font-size: 14px; margin-bottom: 6px; color: #2c3e50; }
    .checkpoint-status { font-size: 12px; color: #34495e; }
    .checkpoint-group { display: flex; align-items: center; gap: 6px; }
    .toolbar input[type="text"] { padding: 2px 4px; }
  </style>
  <script>
    let autoTimer = null;
    async function api(path, method='GET', body=null) {
      const opts = { method, headers: { 'Content-Type': 'application/json' } };
      if (body) opts.body = JSON.stringify(body);
      const res = await fetch(path, opts);
      if (!res.ok) throw new Error(await res.text());
      return await res.json();
    }

    function cardDiv(c, highlightIds) {
      const d = document.createElement('div');
      d.className = 'card' + (highlightIds && highlightIds.includes(c.id) ? ' hl' : '');
      const inner = document.createElement('div');
      inner.className = 'inner s-' + c.suit;
      inner.textContent = c.text;
      d.appendChild(inner);
      return d;
    }

    function render(state) {
      const root = document.getElementById('root');
      const highlightIds = state.last_move ? state.last_move.captured : [];
      let activeSeat = Array.isArray(state.players) ? state.players.findIndex(p => p.name === state.agent) : -1;
      if (activeSeat < 0 && typeof state.agent === 'string') {
        const match = state.agent.match(/player_(\\d+)/);
        if (match) {
          activeSeat = parseInt(match[1], 10);
        }
      }
      const checkpoints = state.checkpoints || {};
      const ckpt0 = checkpoints[0] ?? checkpoints['0'] ?? '';
      const ckpt1 = checkpoints[1] ?? checkpoints['1'] ?? '';
      const ckpt0Input = document.getElementById('ckpt-0');
      if (ckpt0Input && document.activeElement !== ckpt0Input) {
        ckpt0Input.value = ckpt0 || '';
      }
      const ckpt1Input = document.getElementById('ckpt-1');
      if (ckpt1Input && document.activeElement !== ckpt1Input) {
        ckpt1Input.value = ckpt1 || '';
      }
      const ckpt0Status = document.getElementById('ckpt-0-status');
      if (ckpt0Status) {
        ckpt0Status.textContent = ckpt0 ? ckpt0 : 'random';
      }
      const ckpt1Status = document.getElementById('ckpt-1-status');
      if (ckpt1Status) {
        ckpt1Status.textContent = ckpt1 ? ckpt1 : 'random';
      }
      root.innerHTML = '';

      const board = document.createElement('div');
      board.className = 'board';
      root.appendChild(board);

      // Top (player 2)
      const top = document.createElement('div');
      top.className = 'zone';
      const tTitle = document.createElement('div'); tTitle.className='title'; tTitle.textContent = `player_2 (side ${state.players[2].side})`;
      const tCards = document.createElement('div'); tCards.className='cards-row';
      state.players[2].hand.forEach(c => tCards.appendChild(cardDiv(c, highlightIds)));
      top.appendChild(tTitle); top.appendChild(tCards);
      if (activeSeat === 2) { top.classList.add('active-player'); }
      board.appendChild(document.createElement('div')); // spacer
      board.appendChild(top);
      board.appendChild(document.createElement('div')); // spacer

      // Middle row: left, table, right
      const left = document.createElement('div'); left.className='zone col';
      const lTitle = document.createElement('div'); lTitle.className='title'; lTitle.textContent = `player_3 (side ${state.players[3].side})`;
      left.appendChild(lTitle);
      state.players[3].hand.forEach(c => left.appendChild(cardDiv(c, highlightIds)));
      if (activeSeat === 3) { left.classList.add('active-player'); }
      board.appendChild(left);

      const table = document.createElement('div'); table.className='zone';
      const tabTitle = document.createElement('div'); tabTitle.className='title'; tabTitle.textContent = 'Table';
      const tabCards = document.createElement('div'); tabCards.className='cards-row';
      state.table.forEach(c => tabCards.appendChild(cardDiv(c, highlightIds)));
      table.appendChild(tabTitle); table.appendChild(tabCards);
      board.appendChild(table);

      const right = document.createElement('div'); right.className='zone col';
      const rTitle = document.createElement('div'); rTitle.className='title'; rTitle.textContent = `player_1 (side ${state.players[1].side})`;
      right.appendChild(rTitle);
      state.players[1].hand.forEach(c => right.appendChild(cardDiv(c, highlightIds)));
      if (activeSeat === 1) { right.classList.add('active-player'); }
      board.appendChild(right);

      // Bottom (player 0)
      const bottom = document.createElement('div'); bottom.className = 'zone';
      const bTitle = document.createElement('div'); bTitle.className='title'; bTitle.textContent = `player_0 (side ${state.players[0].side})`;
      const bCards = document.createElement('div'); bCards.className='cards-row';
      state.players[0].hand.forEach(c => bCards.appendChild(cardDiv(c, highlightIds)));
      bottom.appendChild(bTitle); bottom.appendChild(bCards);
      if (activeSeat === 0) { bottom.classList.add('active-player'); }
      board.appendChild(document.createElement('div'));
      board.appendChild(bottom);
      board.appendChild(document.createElement('div'));

      const status = document.getElementById('status');
      if (state.done && state.scores) {
        status.textContent = `Finished. Scores: ${JSON.stringify(state.scores)}`;
        if (autoTimer) { clearInterval(autoTimer); autoTimer = null; document.getElementById('auto').checked = false; }
      } else {
        const seatLabel = activeSeat >= 0 ? ` (seat ${activeSeat})` : '';
        status.textContent = `Current: ${state.agent}${seatLabel} | Mode: ${state.mode}`;
      }
    }

    async function refresh() {
      const st = await api('/state');
      render(st);
    }

    async function doStep() {
      const st = await api('/step', 'POST');
      render(st);
    }

    async function doReset() {
      const mode = document.getElementById('mode').value;
      const seed = document.getElementById('seed').value;
      const st = await api('/reset', 'POST', { mode, seed: parseInt(seed || '0') });
      render(st);
    }

    function toggleAuto(chk) {
      const interval = parseInt(document.getElementById('interval').value || '500');
      if (chk.checked) {
        autoTimer = setInterval(async () => {
          try { await doStep(); } catch (e) { console.error(e); clearInterval(autoTimer); autoTimer=null; chk.checked = false; }
        }, Math.max(100, interval));
      } else {
        if (autoTimer) { clearInterval(autoTimer); autoTimer = null; }
      }
    }

    async function applyCheckpoint(team) {
      const input = document.getElementById(`ckpt-${team}`);
      const value = input ? input.value.trim() : '';
      try {
        const st = await api('/checkpoint', 'POST', { team, checkpoint: value || null });
        render(st);
      } catch (err) {
        console.error(err);
        alert(err.message || err);
        await refresh();
      }
    }

    async function clearCheckpoint(team) {
      const input = document.getElementById(`ckpt-${team}`);
      if (input && document.activeElement !== input) {
        input.value = '';
      }
      try {
        const st = await api('/checkpoint', 'POST', { team, checkpoint: null });
        render(st);
      } catch (err) {
        console.error(err);
        alert(err.message || err);
        await refresh();
      }
    }

    window.addEventListener('load', refresh);
  </script>
</head>
<body>
  <div class="toolbar">
    <label>Mode:
      <select id="mode">
        <option value="selfplay">selfplay</option>
        <option value="vs_random">vs_random</option>
      </select>
    </label>
    <label>Seed: <input id="seed" type="number" style="width:100px" /></label>
    <button onclick="doReset()">Reset</button>
    <button onclick="doStep()">Step</button>
    <label><input id="auto" type="checkbox" onchange="toggleAuto(this)"/> Auto</label>
    <label>Interval(ms): <input id="interval" type="number" value="500" style="width:80px"/></label>
    <div class="checkpoint-group">
      <label>Team 0 ckpt: <input id="ckpt-0" type="text" placeholder="path/to_checkpoint.pkl" style="width:220px" /></label>
      <button onclick="applyCheckpoint(0)">Load</button>
      <button onclick="clearCheckpoint(0)">Clear</button>
      <span class="checkpoint-status">Team0: <span id="ckpt-0-status">random</span></span>
    </div>
    <div class="checkpoint-group">
      <label>Team 1 ckpt: <input id="ckpt-1" type="text" placeholder="path/to_checkpoint.pkl" style="width:220px" /></label>
      <button onclick="applyCheckpoint(1)">Load</button>
      <button onclick="clearCheckpoint(1)">Clear</button>
      <span class="checkpoint-status">Team1: <span id="ckpt-1-status">random</span></span>
    </div>
    <div id="status" class="status">Ready</div>
  </div>
  <div id="root"></div>
</body>
</html>
"""


def create_app(team0_checkpoint: Optional[str], team1_checkpoint: Optional[str], mode: str, seed: int, log_dir: Optional[str] = None) -> Flask:
    app = Flask(__name__)
    tlog = TLogger(log_dir=log_dir or ("runs/web/" + time.strftime("%Y-%m-%d-%H-%M-%S")))
    env = make_env(tlog)
    gm = GameManager(tlog=tlog, env=env, mode=mode, seed=seed)
    for team, ckpt in ((0, team0_checkpoint), (1, team1_checkpoint)):
        if ckpt:
            try:
                gm.load_checkpoint(team, ckpt)
            except Exception as exc:
                print(f"WARNING: failed to load checkpoint for team {team} from '{ckpt}': {exc}")
    gm.reset(mode=mode, seed=seed)

    @app.get('/')
    def index() -> Response:
        return Response(HTML_PAGE, mimetype='text/html')

    @app.get('/state')
    def state() -> Response:
        return jsonify(gm.state())

    @app.post('/step')
    def step() -> Response:
        gm.step()
        return jsonify(gm.state())

    @app.post('/reset')
    def reset() -> Response:
        data = request.get_json(silent=True) or {}
        gm.reset(mode=data.get('mode'), seed=data.get('seed'))
        return jsonify(gm.state())

    @app.post('/checkpoint')
    def set_checkpoint() -> Response:
        data = request.get_json(silent=True) or {}
        team = data.get('team')
        checkpoint = data.get('checkpoint')
        try:
            team_idx = int(team)
        except (TypeError, ValueError):
            return Response('Invalid team index', status=400)
        try:
            gm.load_checkpoint(team_idx, checkpoint)
        except Exception as exc:
            return Response(f'Failed to load checkpoint: {exc}', status=400)
        return jsonify(gm.state())

    return app


def main():
    parser = argparse.ArgumentParser(description='Scopa web viewer')
    parser.add_argument('--checkpoint', type=str, default='', help='Legacy shorthand for --team0-checkpoint')
    parser.add_argument('--team0-checkpoint', type=str, default='', help='Checkpoint path controlling seats 0 & 2')
    parser.add_argument('--team1-checkpoint', type=str, default='', help='Checkpoint path controlling seats 1 & 3')
    parser.add_argument('--mode', type=str, default='selfplay', choices=['selfplay', 'vs_random'], help='Self-play or policy team vs random')
    parser.add_argument('--seed', type=int, default=123, help='Seed for randomness')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind')
    parser.add_argument('--port', type=int, default=7860, help='Port to bind')
    args = parser.parse_args()

    team0_ckpt = args.team0_checkpoint or (args.checkpoint or None)
    team1_ckpt = args.team1_checkpoint or None
    app = create_app(team0_ckpt, team1_ckpt, args.mode, args.seed)
    # threaded to allow multiple quick requests during auto-play
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()

