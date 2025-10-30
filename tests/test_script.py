
import numpy as np
from env import env as make_env
from cfr_jax import CFRTrainer

e = make_env(tlogger=None)
print("Env OK:", e is not None)

trainer = CFRTrainer(seed=0, max_infosets=1000, branch_topk=3, subset_cache_size=256)
print("Trainer OK")

# tiny train step
trainer.train(iterations=1, seed=0, batch_size=1, log_every=1, eval_every=None)
print("One iter OK")

