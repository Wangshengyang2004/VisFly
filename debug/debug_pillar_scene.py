"""
Debug pillar scene, describe the scene and the pillar
"""

import numpy as np
import matplotlib.pyplot as plt
import torch as th
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from VisFly.envs.NavigationEnv import NavigationEnv3





if __name__ == "__main__":
    # Debug pillar scene: generate and analyze random targets
    # Configure random spawn area
    random_kwargs = {
        "state_generator": {
            "class": "Uniform",
            "kwargs": [
                {
                    "position": {"mean": [0., 0., 1.], "half": [8., 8., 1.]},
                    "orientation": {"mean": [0., 0., 0.], "half": [0.0, 0.0, 3.1416]},
                    "velocity": {"mean": [0., 0., 0.], "half": [0.1, 0.1, 0.1]},
                    "angular_velocity": {"mean": [0., 0., 0.], "half": [1., 1., 1.]},
                }
            ]
        }
    }
    # Pillar scene configuration
    scene_kwargs = {"path": "../datasets/visfly-beta/configs/scenes/box15_wall_pillar"}

    # Create environment
    env = NavigationEnv3(   
        num_agent_per_scene=1,
        random_kwargs=random_kwargs,
        scene_kwargs=scene_kwargs,
        visual=False,
        requires_grad=False,
        max_episode_steps=1,
        device="cpu"
    )
    env.reset()

    # Starting position (single agent)
    start_pos = env.position.clone().detach()
    N = 1000  # number of random targets
    start_positions = start_pos.repeat(N, 1)

    # Generate targets with up to 10000 attempts
    targets = env.generate_valid_targets(
        start_positions=start_positions,
        num_targets=N,
        min_distance=2.0,
        max_distance=8.0,
        max_attempts=10000
    )

    # Validate generated targets
    valid_mask = env.validate_target_position(targets)
    valid_count = int(valid_mask.sum().item())
    invalid_count = N - valid_count
    print(f"Generated {valid_count} valid and {invalid_count} invalid targets out of {N}")

    # Scatter plot of valid vs invalid targets
    valid_pts = targets[valid_mask].cpu().numpy()
    invalid_pts = targets[~valid_mask].cpu().numpy()
    fig, ax = plt.subplots()
    ax.scatter(valid_pts[:, 0], valid_pts[:, 1], c='green', s=5, label='valid')
    ax.scatter(invalid_pts[:, 0], invalid_pts[:, 1], c='red', s=5, label='invalid')
    ax.set_aspect('equal')
    ax.set_title('Target distribution (green=valid, red=invalid)')
    ax.legend()

    # Annotate pillar center estimated from invalid cluster
    if invalid_count > 0:
        pillar_center = invalid_pts.mean(axis=0)
        ax.scatter(pillar_center[0], pillar_center[1], c='blue', marker='x', s=50, label='pillar center')
        ax.annotate(
            'Pillar',
            xy=(pillar_center[0], pillar_center[1]),
            xytext=(pillar_center[0] + 1, pillar_center[1] + 1),
            arrowprops=dict(arrowstyle='->')
        )

    # Histogram of distances
    plt.figure()
    # Compute distances
    distances = ((targets - start_positions).norm(dim=1)).cpu().numpy()
    plt.hist(distances, bins=50, color='gray', alpha=0.7)
    plt.title('Histogram of target distances')
    plt.xlabel('Distance from start')
    plt.ylabel('Count')

    # Explain why some targets failed
    if invalid_count > 0:
        print(
            "Some targets remain invalid because they lie within the pillar's collision volume, "
            "which cannot be sampled even with 10000 random attempts."
        )

    plt.show()