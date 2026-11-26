"""eval.py with the finger-penetration readout logged: each finished episode's deepest finger-in-plate
depth [m] (FingerPenetration.episode_max, taken at the env's reset), written to $DEPTH_OUT as JSON.
Run with env.termination.max_finger_penetration set too large to trigger (e.g. 1.0) so the readout is
built but never ends an episode."""
import atexit, json, os, runpy, sys

from isaacsimenvs.tasks.cloth.backends.rigid import penetration as pen

LOG = {"episode_max_depth": [], "reset_at_step": [], "episode_len": [], "steps_in_contact": 0, "env_steps": 0}
_reset, _read = pen.FingerPenetration.reset, pen.FingerPenetration.read


def reset(self, env_ids):
    if getattr(self, "_started", False):
        LOG["episode_max_depth"] += [round(float(v), 6) for v in self.episode_max[env_ids].cpu()]
        n = int(self.b.num_envs); step = LOG["env_steps"] // n  # physics steps read so far
        LOG["reset_at_step"] += [step] * len(env_ids)
        LOG["episode_len"] += [step - int(s) for s in self.last_reset[env_ids].cpu()]
    self.last_reset = getattr(self, "last_reset", None)
    if self.last_reset is None:
        import torch
        self.last_reset = torch.zeros(self.b.num_envs, dtype=torch.long, device=self.b.device)
    self.last_reset[env_ids] = LOG["env_steps"] // int(self.b.num_envs)
    if getattr(self, "_started", False):
        _dump()  # Kit may exit without running atexit
    self._started = True
    return _reset(self, env_ids)


def read(self):
    d = _read(self)
    LOG["steps_in_contact"] += int((d > 0).sum()); LOG["env_steps"] += int(d.numel())
    return d


def _dump():
    with open(os.environ["DEPTH_OUT"] + ".tmp", "w") as f:
        json.dump(LOG, f)
    os.replace(os.environ["DEPTH_OUT"] + ".tmp", os.environ["DEPTH_OUT"])


pen.FingerPenetration.reset, pen.FingerPenetration.read = reset, read
atexit.register(_dump)
sys.argv = ["eval.py", *sys.argv[1:]]
runpy.run_module("isaacsimenvs.scripts.evaluation.eval", run_name="__main__")
