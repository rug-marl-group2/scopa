import argparse
import time
import random
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np

try:
    from flask import Flask, request, jsonify, Response
except Exception as e:  # pragma: no cover
    raise SystemExit("Flask is required. Install with: pip install flask")

from tlogger import TLogger
from env import env as make_env
from cfr_jax import CFRTrainer, SavedPolicy


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
    actor: Optional[SavedPolicy]
    mode: str  # 'selfplay' or 'vs_random'
    seed: int
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

    def _select_action(self, seat: int, obs: np.ndarray, mask: np.ndarray) -> int:
        legal = [i for i, m in enumerate(mask) if m == 1]
        if self.mode == 'vs_random' and (seat % 2 == 1):
            return int(random.choice(legal))
        if self.actor is None:
            return int(random.choice(legal))
        a = int(self.actor.act_from_obs(seat, obs))
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
    .zone { padding: 8px; }
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
      board.appendChild(document.createElement('div')); // spacer
      board.appendChild(top);
      board.appendChild(document.createElement('div')); // spacer

      // Middle row: left, table, right
      const left = document.createElement('div'); left.className='zone col';
      const lTitle = document.createElement('div'); lTitle.className='title'; lTitle.textContent = `player_3 (side ${state.players[3].side})`;
      left.appendChild(lTitle);
      state.players[3].hand.forEach(c => left.appendChild(cardDiv(c, highlightIds)));
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
      board.appendChild(right);

      // Bottom (player 0)
      const bottom = document.createElement('div'); bottom.className = 'zone';
      const bTitle = document.createElement('div'); bTitle.className='title'; bTitle.textContent = `player_0 (side ${state.players[0].side})`;
      const bCards = document.createElement('div'); bCards.className='cards-row';
      state.players[0].hand.forEach(c => bCards.appendChild(cardDiv(c, highlightIds)));
      bottom.appendChild(bTitle); bottom.appendChild(bCards);
      board.appendChild(document.createElement('div'));
      board.appendChild(bottom);
      board.appendChild(document.createElement('div'));

      const status = document.getElementById('status');
      if (state.done && state.scores) {
        status.textContent = `Finished. Scores: ${JSON.stringify(state.scores)}`;
        if (autoTimer) { clearInterval(autoTimer); autoTimer = null; document.getElementById('auto').checked = false; }
      } else {
        status.textContent = `Current: ${state.agent} | Mode: ${state.mode}`;
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
    <div id="status" class="status">Ready</div>
  </div>
  <div id="root"></div>
</body>
</html>
"""


def create_app(checkpoint: Optional[str], mode: str, seed: int, log_dir: Optional[str] = None) -> Flask:
    app = Flask(__name__)
    tlog = TLogger(log_dir=log_dir or ("runs/web/" + time.strftime("%Y-%m-%d-%H-%M-%S")))
    env = make_env(tlog)
    actor = None
    if checkpoint:
        try:
            actor = CFRTrainer.load_avg_policy(checkpoint, seed=seed)
        except Exception as e:
            print(f"WARNING: failed to load checkpoint '{checkpoint}': {e}")
    gm = GameManager(tlog=tlog, env=env, actor=actor, mode=mode, seed=seed)
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

    return app


def main():
    parser = argparse.ArgumentParser(description='Scopa web viewer')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to saved checkpoint (.pkl) for policy (optional)')
    parser.add_argument('--mode', type=str, default='selfplay', choices=['selfplay', 'vs_random'], help='Self-play or policy team vs random')
    parser.add_argument('--seed', type=int, default=123, help='Seed for randomness')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind')
    parser.add_argument('--port', type=int, default=7860, help='Port to bind')
    args = parser.parse_args()

    app = create_app(args.checkpoint or None, args.mode, args.seed)
    # threaded to allow multiple quick requests during auto-play
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()

