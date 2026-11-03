"""Asynchronous CPU policy-gradient bandit for glioma augmentation/sampling.

One action controls one epoch; rewards come from a reserved training-internal
glioma holdout, never from model-selection validation or the supplied test set.
This is single-step REINFORCE, not an A3C segmentation agent.
"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import math
import random


ACTIONS = tuple((profile, weight) for profile in ("mild", "standard", "strong") for weight in (1., 1.5, 2.))
LEGACY_POLICY_DEFAULTS = {"glioma_rl": False, "glioma_policy": "standard", "glioma_sampling_weight": 1.,
                          "policy_reward_fraction": 0., "rl_learning_rate": .1, "rl_exploration": .2, "rl_warmup": 3}


def policy_split(samples, fraction, seed, manifest=None):
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("Policy reward fraction must lie between zero and one")
    lookup = {s.image.name: s for s in samples}
    if manifest is None:
        reward = []
        groups = defaultdict(list)
        for sample in samples:
            if sample.tumor == "Glioma":
                groups[sample.plane].append(sample)
        rng = random.Random(seed + 719)
        for plane in sorted(groups):
            group = sorted(groups[plane], key=lambda s: s.image.name)
            if len(group) < 2:
                raise ValueError("Policy reward split needs at least two training glioma slices per plane")
            rng.shuffle(group)
            reward.extend(group[:min(len(group)-1, max(1, round(len(group)*fraction)))])
        reward_names = {s.image.name for s in reward}
        manifest = {"fit": [s.image.name for s in samples if s.image.name not in reward_names],
                    "reward": sorted(reward_names)}
    if set(manifest) != {"fit", "reward"}:
        raise ValueError("Invalid policy split keys")
    fit, reward = manifest["fit"], manifest["reward"]
    if (not fit or not reward or len(fit) != len(set(fit)) or len(reward) != len(set(reward))
            or set(fit) & set(reward) or set(fit) | set(reward) != set(lookup)):
        raise ValueError("Policy fit/reward assignments must be disjoint and cover original training membership")
    if any(lookup[name].tumor != "Glioma" for name in reward):
        raise ValueError("Policy rewards may only use training-internal glioma images")
    return [lookup[name] for name in fit], [lookup[name] for name in reward], manifest


def _update(preferences, baseline, visits, action, reward, learning_rate, learn):
    values = list(preferences)
    if learn:
        soft = action["soft_probabilities"]
        index = action["index"]
        advantage = max(-5., min(5., reward - baseline))
        # Exact log-policy gradient for the epsilon-mixture behavior policy.
        factor = (1 - action["exploration"]) * soft[index] / action["probability"]
        values = [v + learning_rate * advantage * factor * ((j == index) - soft[j])
                  for j, v in enumerate(values)]
        center = sum(values) / len(values)
        values = [max(-6., min(6., v - center)) for v in values]
    counts = list(visits)
    counts[action["index"]] += 1
    return values, .9 * baseline + .1 * reward, counts


class AsyncGliomaPolicy:
    def __init__(self, seed=2026, learning_rate=.1, exploration=.2, warmup=3):
        if not math.isfinite(learning_rate) or learning_rate <= 0 or not 0 <= exploration < 1 or warmup < 0:
            raise ValueError("Invalid policy learning settings")
        self.learning_rate, self.exploration, self.warmup = learning_rate, exploration, warmup
        self.preferences, self.visits = [0.] * len(ACTIONS), [0] * len(ACTIONS)
        self.baseline, self.observations = 0., 0
        self.rng = random.Random(seed + 1831)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="glioma-policy")
        self.pending = None
        self.closed = False

    def flush(self):
        if self.pending is not None:
            self.preferences, self.baseline, self.visits = self.pending.result()
            self.observations += 1
            self.pending = None

    def probabilities(self):
        self.flush()
        weights = [math.exp(p - max(self.preferences)) for p in self.preferences]
        soft = [w / sum(weights) for w in weights]
        return soft, [(1 - self.exploration) * p + self.exploration / len(ACTIONS) for p in soft]

    def choose(self, epoch):
        soft, behavior = self.probabilities()
        index = 3 if epoch <= self.warmup else self.rng.choices(range(len(ACTIONS)), weights=behavior)[0]
        profile, weight = ACTIONS[index]
        return {"index": index, "augmentation": profile, "glioma_sampling_weight": weight,
                "probability": 1. if epoch <= self.warmup else behavior[index], "soft_probabilities": soft, "exploration": self.exploration,
                "warmup": epoch <= self.warmup}

    def observe(self, action, reward, learn=True):
        if self.closed or not math.isfinite(reward):
            raise ValueError("Policy is closed or reward is not finite")
        self.flush()
        reward = max(-5., min(5., float(reward)))
        self.pending = self.executor.submit(_update, self.preferences, self.baseline, self.visits,
                                            dict(action), reward, self.learning_rate, learn)
        return reward

    def state_dict(self):
        self.flush()
        return {"version": 1, "algorithm": "single_step_REINFORCE_bandit", "preferences": list(self.preferences),
                "baseline": self.baseline, "visits": list(self.visits), "observations": self.observations,
                "rng": self.rng.getstate(), "learning_rate": self.learning_rate,
                "exploration": self.exploration, "warmup": self.warmup}

    def load_state_dict(self, state):
        self.flush()
        if (state["version"] != 1 or len(state["preferences"]) != len(ACTIONS)
                or any(state[key] != getattr(self, key) for key in ("learning_rate", "exploration", "warmup"))):
            raise ValueError("Policy checkpoint is incompatible")
        self.preferences, self.visits = list(state["preferences"]), list(state["visits"])
        self.baseline, self.observations = state["baseline"], state["observations"]
        self.rng.setstate(state["rng"])

    def close(self):
        if not self.closed:
            self.flush()
            self.executor.shutdown(wait=True)
            self.closed = True
