#!/usr/bin/env python3

import sys
import os
sys.path.append(os.getcwd())

import torch
import time
import yaml
import numpy as np

from VisFly.utils.policies import extractors
import torch as th
from VisFly.envs.HoverEnv import HoverEnv2
from VisFly.utils.launcher import rl_parser, training_params
from VisFly.utils.algorithms.BPTT import BPTT

args = rl_parser().parse_args()

# Add randomization control arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--enable_yaw_randomization', action='store_true', 
                   help='Enable yaw randomization during training')
parser.add_argument('--enable_target_randomization', action='store_true', 
                   help='Enable target position randomization during training')
parser.add_argument('--enable_agent_randomization', action='store_true', 
                   help='Enable agent starting position randomization during training')
parser.add_argument('--fixed_target', type=float, nargs=3, default=[1.0, 0.0, 1.5],
                   help='Fixed target position [x, y, z] when target randomization is disabled')
parser.add_argument('--fixed_agent_pos', type=float, nargs=3, default=[0.0, 0.0, 1.5],
                   help='Fixed agent starting position [x, y, z] when agent randomization is disabled')

# Parse additional arguments
additional_args, _ = parser.parse_known_args()

save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + "/saved/"
scene_path = "VisFly/datasets/visfly-beta/configs/scenes/garage_empty"

sensor_kwargs = [{
    "sensor_type": "depth",
    "uuid": "depth",
    "resolution": [64, 64],
}]
def main():
    with open(os.path.dirname(os.path.abspath(__file__))+'/cfg/bptt.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Create random_kwargs based on enable flags
    random_kwargs = {"state_generator": {"class": "Uniform", "kwargs": []}}
    
    if additional_args.enable_agent_randomization:
        # Use randomized agent position
        random_kwargs["state_generator"]["kwargs"].append({
            "position": {"mean": [1., 0., 1.5], "half": [1.0, 1.0, 0.5]}
        })
    else:
        # Use fixed agent position
        random_kwargs["state_generator"]["kwargs"].append({
            "position": {"mean": additional_args.fixed_agent_pos, "half": [0.0, 0.0, 0.0]}
        })
    
    if additional_args.enable_yaw_randomization:
        # Add yaw randomization
        random_kwargs["state_generator"]["kwargs"].append({
            "orientation": {"mean": [0., 0., 0.], "half": [0.1, 0.1, 0.1]}
        })
    else:
        # Calculate yaw to face target and set upright orientation
        target_pos = additional_args.fixed_target
        agent_pos = additional_args.fixed_agent_pos
        direction = [target_pos[0] - agent_pos[0], target_pos[1] - agent_pos[1], 0]  # Ignore Z for yaw
        yaw = np.arctan2(direction[1], direction[0])
        random_kwargs["state_generator"]["kwargs"].append({
            "orientation": {"mean": [0., 0., yaw], "half": [0.0, 0.0, 0.0]}  # Upright, facing target
        })

    # if train mode, train the model
    if args.train:

        # Set target based on randomization flag
        if additional_args.enable_target_randomization:
            # Use default target (will be randomized by environment)
            target = None
        else:
            # Use fixed target
            target = torch.tensor(additional_args.fixed_target)

        env = HoverEnv2(
            **config["env"],
            random_kwargs=random_kwargs,
            sensor_kwargs=sensor_kwargs,
            target=target
        )
        
        # Example of using target validation (uncomment to use)
        # if additional_args.enable_target_randomization:
        #     # Generate valid targets using the new validation function
        #     start_positions = env.position  # Get current agent positions
        #     valid_targets = env.generate_valid_targets(
        #         start_positions=start_positions,
        #         num_targets=env.num_envs,
        #         min_distance=2.0,
        #         max_distance=8.0,
        #         max_attempts=100
        #     )
        #     # Set the valid targets
        #     success = env.set_valid_targets(valid_targets)
        #     if success:
        #         print("Successfully set valid targets")
        #     else:
        #         print("Warning: Some targets are invalid")

        # Load pretrained model if provided
        if args.weight is not None:
            model = BPTT.load(path=os.path.join(save_folder + args.weight))
        else:
            model = BPTT(
                env=env,
                seed=args.seed,
                comment=args.comment,
                **config["algorithm"]
            )

        model.learn(**config["learn"])
        model.save()

    # Testing mode with a trained weight
    else:
        test_model_path = save_folder + args.weight
        
        # Set target for test mode
        if additional_args.enable_target_randomization:
            target = None
        else:
            target = torch.tensor(additional_args.fixed_target)
        
        # Add render settings for test environment
        test_scene_kwargs = config["eval_env"].get("scene_kwargs", {}).copy()
        test_scene_kwargs["render_settings"] = {
            "mode": "fix",
            "view": "custom",
            "resolution": [1080, 1920],
            "position": torch.tensor([[7., 6.8, 5.5], [7, 4.8, 4.5]]),
            "trajectory": True,
        }
        
        from tst import Test
        eval_env_config = config["eval_env"].copy()
        eval_env_config["scene_kwargs"] = test_scene_kwargs
        eval_env_config["random_kwargs"] = random_kwargs
        eval_env_config["target"] = target
        env = HoverEnv2(**eval_env_config)
        
        # Initialize BPTT model for testing
        model = BPTT(
            env=env,
            **config["algorithm"]
        )
        
        print(f"Loading model from: {test_model_path}")
        model.load(test_model_path)
        
        # Wrap model for test
        class ModelWrapper:
            def __init__(self, policy):
                self.policy = policy
                self.env = env
        wrapped = ModelWrapper(model.policy)
        
        test_handle = Test(
            env,
            wrapped,
            "hover_test",
            os.path.dirname(os.path.realpath(__file__)) + "/saved/test"
        )
        test_args = config["test"].copy()
        test_args.pop("is_render", None)  # Remove if present
        test_handle.test(**test_args)


if __name__ == "__main__":
    main()
