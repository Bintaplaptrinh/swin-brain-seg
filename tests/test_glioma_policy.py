import json
from pathlib import Path
import random
from types import SimpleNamespace
import threading

import numpy as np
from PIL import Image
import pytest
import torch

from src.datasets.brisc import BRISCDataset, Sample
from src.trainers.glioma_policy import AsyncGliomaPolicy, policy_split
from src.trainers.baseline import loader, parse_args, checkpoint_setting, prepare_resume, MODEL_FIELDS, TRAIN_FIELDS


def samples():
    return [Sample(Path(f"{tumor}_{plane}_{i}.jpg"), Path("mask.png"), tumor, plane)
            for tumor in ("Glioma", "Meningioma", "Pituitary")
            for plane in ("Axial", "Coronal", "Sagittal") for i in range(10)]


def test_reward_holdout_is_disjoint_glioma_only_and_reproducible():
    pool = samples()
    fit, reward, manifest = policy_split(pool, .1, 2026)
    assert len(reward) == 3 and len(fit) == 87
    assert {s.tumor for s in reward} == {"Glioma"}
    assert not {s.image.name for s in fit} & {s.image.name for s in reward}
    assert (fit, reward, manifest) == policy_split(pool, .1, 2026)
    assert (fit, reward, manifest) == policy_split(pool, .1, 2026, manifest)
    manifest["reward"][0] = fit[0].image.name
    with pytest.raises(ValueError):
        policy_split(pool, .1, 2026, manifest)


def test_positive_policy_reward_increases_selected_action_probability_and_restores_rng(tmp_path):
    policy = AsyncGliomaPolicy(warmup=0)
    restored = AsyncGliomaPolicy(warmup=0)
    try:
        action = policy.choose(1)
        policy.observe(action, 4)
        state = policy.state_dict()
        assert policy.probabilities()[1][action["index"]] > action["probability"]
        torch.save(state, tmp_path / "policy.pt")
        restored.load_state_dict(torch.load(tmp_path / "policy.pt", weights_only=True))
        assert [policy.choose(e) for e in range(2, 10)] == [restored.choose(e) for e in range(2, 10)]
    finally:
        policy.close()
        restored.close()


def test_policy_update_executes_asynchronously_on_cpu(monkeypatch):
    from src.trainers import glioma_policy
    original = glioma_policy._update
    started, release = threading.Event(), threading.Event()
    thread_names = []

    def controlled(*args):
        thread_names.append(threading.current_thread().name)
        started.set()
        assert release.wait(3)
        return original(*args)

    monkeypatch.setattr(glioma_policy, "_update", controlled)
    policy = AsyncGliomaPolicy(warmup=0)
    try:
        policy.observe(policy.choose(1), 1)
        assert started.wait(3)
        assert not policy.pending.done()
        assert thread_names[0].startswith("glioma-policy")
    finally:
        release.set()
        policy.close()


def test_glioma_only_augmentation_leaves_other_tumor_images_deterministic(tmp_path):
    rng = np.random.default_rng(3)
    image = rng.integers(20, 230, (40, 64), dtype=np.uint8)
    mask = np.zeros((40, 64), dtype=np.uint8)
    mask[12:28, 20:42] = 255
    ip, mp = tmp_path / "image.png", tmp_path / "mask.png"
    Image.fromarray(image).save(ip)
    Image.fromarray(mask).save(mp)
    for tumor in ("Glioma", "Meningioma", "Pituitary"):
        sample = Sample(ip, mp, tumor, "Axial")
        plain = BRISCDataset([sample], size=64, augment=False)[0]
        random.seed(2026)
        np.random.seed(2026)
        augmented = BRISCDataset([sample], size=64, augment=True, augmentation="glioma", glioma_policy="strong")[0]
        if tumor == "Glioma":
            assert not torch.equal(plain[0], augmented[0])
            assert set(augmented[1].unique().tolist()) <= {-1, 0, 1}
        else:
            torch.testing.assert_close(plain[0], augmented[0], atol=0, rtol=0)
            torch.testing.assert_close(plain[1], augmented[1], atol=0, rtol=0)


def test_weighted_sampler_changes_glioma_frequency_without_changing_epoch_length():
    pool = samples()
    args = SimpleNamespace(size=64, augmentation="glioma", batch_size=4, workers=0, device="cpu")
    batches = loader(pool, args, torch.Generator().manual_seed(2026), training=True, sampling_weight=2)
    draws = [i for _ in range(60) for i in batches.sampler]
    frequency = sum(pool[i].tumor == "Glioma" for i in draws) / len(draws)
    assert .46 < frequency < .54
    assert len(batches.sampler) == len(pool)


def test_new_defaults_and_old_checkpoint_compatibility():
    args = parse_args(["--device", "cpu"])
    assert args.variant == "biref-ema" and args.glioma_rl
    assert args.augmentation == "glioma" and args.policy_reward_fraction == .1
    assert args.ema_skip_stages == [3, 4, 5] and args.dw_dilation == 1
    assert checkpoint_setting({"variant": "biref"}, "glioma_rl") is False
    assert checkpoint_setting({"variant": "biref"}, "ema_skip_stages") == [3, 4, 5]
    old = parse_args(["--device", "cpu", "--variant", "biref"])
    assert not old.glioma_rl and old.augmentation == "robust" and old.policy_reward_fraction == 0


def test_resume_rejects_a_best_companion_with_different_policy_holdout(tmp_path):
    config = {key: 1 for key in MODEL_FIELDS + TRAIN_FIELDS}
    best = {"epoch": 1, "best_epoch": 1, "best_score": .7, "splits": {}, "config": config,
            "policy_split": {"fit": ["a"], "reward": ["b"]}}
    torch.save(best, tmp_path / "best.pt")
    checkpoint = {**best, "epoch": 2, "policy_split": {"fit": ["b"], "reward": ["a"]}}
    with pytest.raises(ValueError, match="does not match"):
        prepare_resume(SimpleNamespace(output=tmp_path, resume=tmp_path / "last.pt"), checkpoint)


@pytest.mark.parametrize("options", [["--policy-reward-fraction", "0"], ["--rl-exploration", "1"],
                                     ["--rl-learning-rate", "nan"], ["--augmentation", "robust"],
                                     ["--ema-skip-stages", "6"]])
def test_invalid_controller_configuration(options):
    with pytest.raises(SystemExit):
        parse_args(["--device", "cpu", *options])
