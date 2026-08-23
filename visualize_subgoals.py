"""Visualize baseline and BiMixer intentions as nearest offline states."""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate_agents import (
    create_agent,
    infer_task_latent,
    resolve_checkpoint,
)
from utils.env_utils import make_env_and_datasets


METHOD_LABELS = {
    "baseline": "Single-intention baseline",
    "bimixer": "Bidirectional Mixer",
}
SLOT_COLORS = ["#e63946", "#f4a261", "#9b5de5", "#00a896", "#457b9d"]


def close_env(env):
    """Close OGBench's additional renderer before closing the Gym environment."""
    renderer = getattr(env.unwrapped, "custom_renderer", None)
    if renderer is not None:
        renderer.close()
    env.close()


class LatentStateDecoder:
    """Map an FB latent to the offline state with maximum cosine similarity."""

    def __init__(self, agent, dataset, candidate_count, batch_size, seed):
        rng = np.random.default_rng(seed)
        candidate_count = min(candidate_count, dataset.size)
        candidate_idxs = np.sort(
            rng.choice(dataset.size, size=candidate_count, replace=False)
        )
        self.xy = np.asarray(dataset["qpos"])[candidate_idxs, :2]
        observations = np.asarray(dataset["observations"])[candidate_idxs]

        representations = []
        for start in range(0, candidate_count, batch_size):
            batch = observations[start : start + batch_size]
            backward = agent.network.select("backward_repr")(jnp.asarray(batch))
            normalized = agent.normalize_z(backward)
            representations.append(np.asarray(normalized))
        self.representations = np.concatenate(representations, axis=0)
        self.latent_dim = self.representations.shape[-1]

    def decode(self, plan):
        plan = np.asarray(plan)
        similarities = plan @ self.representations.T / self.latent_dim
        nearest_idxs = np.argmax(similarities, axis=-1)
        nearest_similarities = similarities[np.arange(len(plan)), nearest_idxs]
        return self.xy[nearest_idxs], nearest_similarities


@jax.jit
def policy_step(agent, observation, task_latent):
    """Return the deterministic full plan and action used for its first slot."""
    observation = jnp.asarray(observation)
    task_latent = jnp.asarray(task_latent)
    high_dist = agent.network.select("high_actor")(
        observation,
        task_latent,
        goal_encoded=True,
        temperature=1.0,
    )
    plan = agent.normalize_z(high_dist.mode())
    if plan.ndim == 1:
        plan = plan[None, :]

    low_dist = agent.network.select("actor")(
        observation,
        plan[0],
        goal_encoded=True,
        temperature=1.0,
    )
    action = jnp.clip(low_dist.mode(), -1, 1)
    return plan, action


def get_agent_xy(env, observation):
    if hasattr(env.unwrapped, "data"):
        return np.asarray(env.unwrapped.data.qpos[:2]).copy()
    return np.asarray(observation[:2]).copy()


def run_rollout(
    method,
    checkpoint,
    frozen_path,
    env_name,
    task_id,
    seed,
    plan_interval,
    candidate_count,
    decoder_batch_size,
    num_zero_shot_samples,
):
    """Run one deterministic episode and record periodically decoded plans."""
    env, train_dataset, val_dataset = make_env_and_datasets(env_name, add_info=True)
    env.unwrapped._add_noise_to_goal = False
    dataset = val_dataset if val_dataset is not None else train_dataset
    example_batch = train_dataset.sample(1)
    agent, _ = create_agent(
        method,
        checkpoint,
        example_batch,
        frozen_path=frozen_path,
        seed=seed,
    )

    np.random.seed(seed)
    env.reset(seed=seed)
    observation, info = env.reset(options={"task_id": task_id})
    task_latent = infer_task_latent(
        agent,
        env_name,
        env,
        dataset,
        num_zero_shot_samples,
    )
    decoder = LatentStateDecoder(
        agent,
        dataset,
        candidate_count=candidate_count,
        batch_size=decoder_batch_size,
        seed=seed,
    )
    trajectory = [get_agent_xy(env, observation)]
    plans = []
    done = False
    step = 0
    final_info = info
    while not done:
        plan, action = policy_step(agent, observation, task_latent)
        plan, action = np.asarray(plan), np.asarray(action)
        slots = np.arange(1, len(plan) + 1)
        planner_diagnostics = {
            "plan_version": step + 1,
            "active_slot": 0,
            "current_similarity": np.nan,
        }
        planner_event = "stateless_replan"
        if step % plan_interval == 0:
            decoded_xy, similarities = decoder.decode(plan)
            plans.append(
                {
                    "step": step,
                    "origin_xy": trajectory[-1].copy(),
                    "decoded_xy": decoded_xy,
                    "similarities": similarities,
                    "slots": slots,
                    "plan_version": planner_diagnostics["plan_version"],
                    "active_slot": planner_diagnostics["active_slot"],
                    "active_similarity": planner_diagnostics["current_similarity"],
                    "event": planner_event,
                }
            )

        observation, _, terminated, truncated, final_info = env.step(action)
        trajectory.append(get_agent_xy(env, observation))
        done = terminated or truncated
        step += 1

    result = {
        "method": method,
        "task_id": task_id,
        "success": float(final_info.get("success", 0.0)),
        "trajectory": np.asarray(trajectory),
        "plans": plans,
        "goal_xy": np.asarray(env.unwrapped.cur_goal_xy).copy(),
        "maze_xy": np.asarray(dataset["qpos"])[:, :2],
    }
    close_env(env)
    return result


def select_snapshots(plans, count):
    if len(plans) <= count:
        return plans
    idxs = np.linspace(0, len(plans) - 1, count).round().astype(int)
    return [plans[idx] for idx in idxs]


def draw_maze(ax, maze_xy):
    max_points = 50_000
    if len(maze_xy) > max_points:
        idxs = np.linspace(0, len(maze_xy) - 1, max_points).astype(int)
        maze_xy = maze_xy[idxs]
    ax.scatter(maze_xy[:, 0], maze_xy[:, 1], s=1, c="#d8d4ca", alpha=0.35)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_plan(ax, result, plan):
    draw_maze(ax, result["maze_xy"])
    step = min(plan["step"], len(result["trajectory"]) - 1)
    trajectory = result["trajectory"][: step + 1]
    ax.plot(trajectory[:, 0], trajectory[:, 1], color="#264653", linewidth=2)
    ax.scatter(*trajectory[0], s=55, c="#2a9d8f", marker="o", zorder=4)
    ax.scatter(*result["goal_xy"], s=130, c="#e9c46a", marker="*", edgecolor="black", zorder=5)

    chain = np.concatenate([plan["origin_xy"][None], plan["decoded_xy"]], axis=0)
    ax.plot(chain[:, 0], chain[:, 1], linestyle="--", color="#555555", alpha=0.8)
    ax.scatter(*plan["origin_xy"], s=50, c="#1d3557", marker="x", zorder=5)
    for slot, xy, similarity in zip(
        plan["slots"], plan["decoded_xy"], plan["similarities"]
    ):
        color = SLOT_COLORS[(slot - 1) % len(SLOT_COLORS)]
        ax.scatter(*xy, s=75, c=color, edgecolor="white", linewidth=0.8, zorder=6)
        ax.annotate(
            f"z{slot} ({similarity:.2f})",
            xy,
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color=color,
            weight="bold",
        )
    ax.set_title(
        f"step {plan['step']} | plan {plan['plan_version']} | {plan['event']}",
        fontsize=10,
    )


def save_comparison(results, output_path, snapshot_count):
    snapshots = {
        method: select_snapshots(result["plans"], snapshot_count)
        for method, result in results.items()
    }
    num_rows = max(len(value) for value in snapshots.values())
    fig, axes = plt.subplots(
        num_rows,
        len(results),
        figsize=(7 * len(results), 5.5 * num_rows),
        squeeze=False,
    )
    for column, (method, result) in enumerate(results.items()):
        for row in range(num_rows):
            ax = axes[row, column]
            if row >= len(snapshots[method]):
                ax.axis("off")
                continue
            plot_plan(ax, result, snapshots[method][row])
            if row == 0:
                ax.text(
                    0.5,
                    1.12,
                    f"{METHOD_LABELS[method]} | success={result['success']:.0f}",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=14,
                    weight="bold",
                )

    fig.suptitle(
        "Decoded high-level intentions\n"
        "Each zi is the offline state with maximum cosine similarity to the predicted FB latent",
        fontsize=15,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_plan_csv(results, output_path):
    fieldnames = [
        "method",
        "task_id",
        "success",
        "step",
        "slot",
        "origin_x",
        "origin_y",
        "decoded_x",
        "decoded_y",
        "cosine_similarity",
        "plan_version",
        "active_slot",
        "active_similarity",
        "event",
    ]
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for method, result in results.items():
            for plan in result["plans"]:
                for slot, xy, similarity in zip(
                    plan["slots"], plan["decoded_xy"], plan["similarities"]
                ):
                    writer.writerow(
                        {
                            "method": method,
                            "task_id": result["task_id"],
                            "success": result["success"],
                            "step": plan["step"],
                            "slot": slot,
                            "origin_x": plan["origin_xy"][0],
                            "origin_y": plan["origin_xy"][1],
                            "decoded_x": xy[0],
                            "decoded_y": xy[1],
                            "cosine_similarity": similarity,
                            "plan_version": plan["plan_version"],
                            "active_slot": plan["active_slot"] + 1,
                            "active_similarity": plan["active_similarity"],
                            "event": plan["event"],
                        }
                    )


def parse_tasks(value):
    if value == "all":
        return "all"
    return [int(task.strip()) for task in value.split(",") if task.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", default="/home/makanin/rl/medium/params.pkl")
    parser.add_argument(
        "--bimixer-checkpoint",
        type=Path,
        required=True,
        help="Direct path to the BiMixer high-level policy .pkl checkpoint.",
    )
    parser.add_argument("--frozen-path", default="/home/makanin/rl/medium")
    parser.add_argument("--env-name", default="ogbench-antmaze-medium-navigate-v0")
    parser.add_argument("--tasks", type=parse_tasks, default=parse_tasks("all"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plan-interval", type=int, default=25)
    parser.add_argument("--snapshot-count", type=int, default=4)
    parser.add_argument("--candidate-count", type=int, default=25_000)
    parser.add_argument("--decoder-batch-size", type=int, default=4096)
    parser.add_argument("--num-zero-shot-samples", type=int, default=100_000)
    parser.add_argument("--output-dir", type=Path, default=Path("subgoal_visualizations"))
    return parser.parse_args()


def main():
    args = parse_args()
    baseline_checkpoint = resolve_checkpoint(args.baseline_checkpoint)
    bimixer_checkpoint = args.bimixer_checkpoint.expanduser().resolve()
    if bimixer_checkpoint.suffix != ".pkl":
        raise ValueError(
            f"--bimixer-checkpoint must point directly to a .pkl file, got: {bimixer_checkpoint}"
        )
    if not bimixer_checkpoint.is_file():
        raise FileNotFoundError(
            f"BiMixer high-level policy checkpoint not found: {bimixer_checkpoint}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.tasks == "all":
        probe_env = make_env_and_datasets(args.env_name, env_only=True, add_info=True)
        task_infos = probe_env.unwrapped.task_infos
        tasks = list(range(1, len(task_infos) + 1))
        close_env(probe_env)
    else:
        tasks = args.tasks

    for task_index, task_id in enumerate(tasks, start=1):
        prefix = f"Task {task_id} ({task_index}/{len(tasks)})"
        print(f"{prefix}: running baseline rollout...", flush=True)
        baseline_result = run_rollout(
            "baseline",
            baseline_checkpoint,
            args.frozen_path,
            args.env_name,
            task_id,
            args.seed,
            args.plan_interval,
            args.candidate_count,
            args.decoder_batch_size,
            args.num_zero_shot_samples,
        )
        print(f"{prefix}: running BiMixer rollout...", flush=True)
        bimixer_result = run_rollout(
            "bimixer",
            bimixer_checkpoint,
            args.frozen_path,
            args.env_name,
            task_id,
            args.seed,
            args.plan_interval,
            args.candidate_count,
            args.decoder_batch_size,
            args.num_zero_shot_samples,
        )
        print(f"{prefix}: rendering comparison...", flush=True)
        results = {"baseline": baseline_result, "bimixer": bimixer_result}
        image_path = args.output_dir / f"task_{task_id}_comparison.png"
        csv_path = args.output_dir / f"task_{task_id}_subgoals.csv"
        save_comparison(results, image_path, args.snapshot_count)
        save_plan_csv(results, csv_path)
        print(f"Task {task_id}: {image_path}")
        print(f"Task {task_id}: {csv_path}")


if __name__ == "__main__":
    main()
