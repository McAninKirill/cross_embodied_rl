"""Evaluate trained FB pi-Switch policies without updating their weights."""

import argparse
import csv
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import flax
import jax
import numpy as np

from agents.fbpiswitch import FBpiSwitchAgent, get_config as get_baseline_config
from agents.fbpiswitch_bimixer import FBpiSwitchBiMixerAgent, get_config as get_bimixer_config
from utils.env_utils import make_env_and_datasets, relabel_dataset
from utils.evaluation import evaluate


METHODS = {
    "baseline": (FBpiSwitchAgent, get_baseline_config),
    "bimixer": (FBpiSwitchBiMixerAgent, get_bimixer_config),
}


def resolve_checkpoint(path):
    """Resolve and validate a direct path to a .pkl checkpoint file."""
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() != ".pkl":
        raise ValueError(
            f"Checkpoint must be specified as a direct path to a .pkl file, got: {path}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {path}")
    return path


def load_config(method, checkpoint, frozen_path=None):
    """Build the current config and overlay the config saved with the weights."""
    _, config_fn = METHODS[method]
    config = config_fn()
    flags_path = checkpoint.parent / "flags.json"
    if flags_path.exists():
        with flags_path.open() as file:
            saved_flags = json.load(file)
        config.update(saved_flags.get("agent", {}))

    # The selected implementation, not stale checkpoint metadata, owns these fields.
    if method == "baseline":
        config.agent_name = "fbpiswitch"
        config.dataset_class = "HGCDataset"
    else:
        config.agent_name = "fbpiswitch_bimixer"
        config.dataset_class = "HGCMultiGoalDataset"

    if frozen_path is not None:
        config.frozen_path = str(Path(frozen_path).expanduser().resolve())
    return config


def restore_network_params(agent, checkpoint):
    """Restore every network module while ignoring optimizer compatibility."""
    with checkpoint.open("rb") as file:
        saved = pickle.load(file)
    saved_params = saved["agent"]["network"]["params"]

    params = flax.core.unfreeze(agent.network.params)
    missing = set(params) - set(saved_params)
    unexpected = set(saved_params) - set(params)
    if missing or unexpected:
        raise ValueError(
            "Checkpoint architecture does not match the selected method. "
            f"Missing modules: {sorted(missing)}; unexpected modules: {sorted(unexpected)}"
        )

    for module_name in params:
        params[module_name] = flax.serialization.from_state_dict(
            params[module_name], saved_params[module_name]
        )
    return agent.replace(network=agent.network.replace(params=flax.core.freeze(params)))


def create_agent(method, checkpoint, example_batch, frozen_path=None, seed=0):
    """Create an agent with the architecture and weights of a saved experiment."""
    agent_class, _ = METHODS[method]
    config = load_config(method, checkpoint, frozen_path=frozen_path)
    agent = agent_class.create(seed, example_batch, config)
    return restore_network_params(agent, checkpoint), config


def infer_task_latent(agent, env_name, env, dataset, num_samples):
    """Infer the zero-shot task latent from the relabeled offline dataset."""
    relabeled = relabel_dataset(env_name, env, dataset)
    num_samples = min(num_samples, relabeled.size)
    batch = relabeled.sample(num_samples, idxs=np.arange(num_samples))
    return np.asarray(agent.infer_latent(batch))


def evaluate_all_tasks(
    agent,
    env_name,
    env,
    zero_shot_dataset,
    seeds,
    episodes,
    num_zero_shot_samples,
    temperature=0.0,
):
    """Evaluate all environment tasks and return one result row per seed."""
    task_infos = (
        env.unwrapped.task_infos
        if hasattr(env.unwrapped, "task_infos")
        else env.task_infos
    )
    rows = []
    for seed in seeds:
        np.random.seed(seed)
        env.reset(seed=seed)
        row = {"seed": seed}
        task_successes = []
        for task_id in range(1, len(task_infos) + 1):
            env.reset(options={"task_id": task_id})
            latent = infer_task_latent(
                agent,
                env_name,
                env,
                zero_shot_dataset,
                num_zero_shot_samples,
            )
            info, _, _ = evaluate(
                agent=agent,
                env=env,
                task_id=task_id,
                inferred_latent=latent,
                num_eval_episodes=episodes,
                num_video_episodes=0,
                eval_temperature=temperature,
            )
            success = float(info["success"])
            row[f"task_{task_id}_success"] = success
            task_successes.append(success)
        row["overall_success"] = float(np.mean(task_successes))
        rows.append(row)
    return rows


def summarize(rows):
    """Aggregate evaluation rows across random seeds."""
    metric_names = [name for name in rows[0] if name != "seed"]
    return {
        name: {
            "mean": float(np.mean([row[name] for row in rows])),
            "std": float(np.std([row[name] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
        }
        for name in metric_names
    }


def write_results(rows, summary, output_dir, method, checkpoint, args):
    """Write machine-readable per-seed and aggregate results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result_name = method
    csv_path = output_dir / f"{result_name}_results.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "method": method,
        "checkpoint": str(checkpoint),
        "env_name": args.env_name,
        "episodes_per_task": args.episodes,
        "seeds": args.seeds,
        "num_zero_shot_samples": args.num_zero_shot_samples,
        "summary": summary,
    }
    json_path = output_dir / f"{result_name}_summary.json"
    with json_path.open("w") as file:
        json.dump(report, file, indent=2)
    return csv_path, json_path


def parse_seeds(value):
    return [int(seed.strip()) for seed in value.split(",") if seed.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Direct path to the policy weights/checkpoint .pkl file.",
    )
    parser.add_argument("--frozen-path", default="/home/makanin/rl/medium")
    parser.add_argument("--env-name", default="ogbench-antmaze-medium-navigate-v0")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per task and seed.")
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("0,1,2,3,4"))
    parser.add_argument("--num-zero-shot-samples", type=int, default=100_000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_results"))
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint)
    env, train_dataset, val_dataset = make_env_and_datasets(args.env_name, add_info=True)
    env.unwrapped._add_noise_to_goal = False
    zero_shot_dataset = val_dataset if val_dataset is not None else train_dataset
    example_batch = train_dataset.sample(1)

    agent, config = create_agent(
        args.method,
        checkpoint,
        example_batch,
        frozen_path=args.frozen_path,
        seed=args.seeds[0],
    )
    rows = evaluate_all_tasks(
        agent,
        args.env_name,
        env,
        zero_shot_dataset,
        args.seeds,
        args.episodes,
        args.num_zero_shot_samples,
        temperature=args.temperature,
    )
    summary = summarize(rows)
    csv_path, json_path = write_results(
        rows, summary, args.output_dir, args.method, checkpoint, args
    )
    env.close()

    print(f"Checkpoint: {checkpoint}")
    for name, stats in summary.items():
        print(f"{name}: {stats['mean']:.3f} +/- {stats['std']:.3f}")
    print(f"Per-seed results: {csv_path}")
    print(f"Summary: {json_path}")


if __name__ == "__main__":
    main()