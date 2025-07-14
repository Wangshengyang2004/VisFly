#!/usr/bin/env python3

import sys
import os
import torch as th
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from VisFly.envs.NavigationEnv import NavigationEnv2
import cv2 as cv
from habitat_sim.sensor import SensorType
from VisFly.utils.maths import Quaternion

random_kwargs = {
    "state_generator":
        {
            "class": "Uniform",
            "kwargs": [
                {"position": {"mean": [1., 0., 1.5], "half": [2.0, 2.0, 1.0]}},
            ]
        }
}

dynamics_kwargs = {
    "action_type": "bodyrate",
    "ori_output_type": "quaternion",
    "dt": 0.005,
    "ctrl_dt": 0.03,
    "ctrl_delay": True,
    "comm_delay": 0.09,
    "action_space": (-1, 1),
    "integrator": "euler",
    "drag_random": 0,
}

# Try to create environment with scene, fallback to non-visual if needed
try:
    scene_path = "VisFly/datasets/visfly-beta/configs/garage_empty"
    sensor_kwargs = [{
        "sensor_type": SensorType.COLOR,
        "uuid": "depth",
        "resolution": [128, 128],
        "position": [0,0.2,0.],
    }]
    scene_kwargs = {
        "path": scene_path,
        "render_settings": {
            "mode": "fix",
            "view": "custom",
            "resolution": [1080, 1920],
            "position": th.tensor([[7., 6.8, 5.5], [7, 4.8, 4.5]]),
            "line_width": 6.,
            "trajectory": True,
        }
    }
    visual = True
except Exception as e:
    print(f"\033[91mScene loading failed, using non-visual mode: {e}\033[0m")
    sensor_kwargs = []
    scene_kwargs = {}
    visual = False

num_agent = 4
env = NavigationEnv2(
    visual=visual,
    num_scene=1,
    num_agent_per_scene=num_agent,
    random_kwargs=random_kwargs,
    dynamics_kwargs=dynamics_kwargs,
    scene_kwargs=scene_kwargs,
    sensor_kwargs=sensor_kwargs
)

print("Environment created successfully!")
env.reset()
print("Environment reset successfully!")
env.reset_env_by_id(0)
print("Environment reset by ID successfully!")
print("Reset scene test completed!")
print("Test completed!")
