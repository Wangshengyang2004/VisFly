#!/usr/bin/env python3

import sys
import os
import time
import torch
import numpy as np

# Ensure project root is in path
sys.path.append(os.getcwd())

from VisFly.utils.policies import extractors
from VisFly.utils.algorithms.BPTT import BPTT
from VisFly.utils import savers
from VisFly.envs.LandingEnv import LandingEnv
from VisFly.utils.launcher import rl_parser, training_params
from VisFly.utils.type import Uniform

args = rl_parser().parse_args()

""" SAVED HYPERPARAMETERS """
# Number of parallel environments (agents)
training_params["num_env"] = 48
# Total learning steps
training_params["learning_step"] = 1e6
# Comments and seed
training_params["comment"] = args.comment
training_params["seed"] = args.seed
# Episode length
training_params["max_episode_steps"] = 256
# Learning rate for BPTT
training_params["learning_rate"] = 1e-3
# BPTT horizon (how many steps to backprop through)
training_params["horizon"] = training_params["max_episode_steps"]

# Directory where to save checkpoints and logs
save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + "/saved/"

# Scene configuration
scene_path = "VisFly/datasets/visfly-beta/configs/scenes/garage_landing"

# Random initialization for environment resets
random_kwargs = {
    "state_generator": {
        "class": "Uniform",
        "kwargs": [
            {
                "position": {"mean": [2., 0., 1.5], "half": [.5, .5, .2]},
            }
        ]
    }
}

def main():
    # Training mode
    if args.train:
        # Create LandingEnv with gradients enabled for BPTT
        env = LandingEnv(
            num_agent_per_scene=int(training_params["num_env"]),
            random_kwargs=random_kwargs,
            visual=True,
            device="cpu",
            max_episode_steps=int(training_params["max_episode_steps"]),
            requires_grad=True,  # Enable gradients for BPTT algorithm
            scene_kwargs={
                "path": scene_path,
            },
        )
        env.reset()
        
        # Create separate evaluation environment (without gradients for efficiency)
        print("Creating evaluation environment...")
        eval_env = LandingEnv(
            num_agent_per_scene=int(training_params["num_env"]),
            random_kwargs=random_kwargs,
            visual=True,
            device="cpu",
            max_episode_steps=int(training_params["max_episode_steps"]),
            requires_grad=False,  # No gradients needed for evaluation
            scene_kwargs={
                "path": scene_path,
            },
        )
        eval_env.reset()
        
        # Load pretrained model if provided
        if args.weight is not None:
            model = BPTT.load(path=os.path.join(save_folder + args.weight))
        else:
            # Instantiate BPTT algorithm
            model = BPTT(
                env=env,
                policy="MultiInputPolicy",
                policy_kwargs=dict(
                    features_extractor_class=extractors.StateTargetImageExtractor,
                    features_extractor_kwargs={
                        "net_arch": {
                            "color": {
                                "layer": [128],
                            },
                            "state": {
                                "layer": [128, 64],
                            },
                            "target": {
                                "layer": [128, 64],
                            }
                        }
                    },
                    net_arch=dict(
                        pi=[64, 64],
                        qf=[64, 64]),
                    activation_fn=torch.nn.ReLU,
                    optimizer_kwargs=dict(weight_decay=1e-5)
                ),
                learning_rate=training_params["learning_rate"],
                horizon=int(training_params["horizon"]),
                gamma=training_params.get("gamma", 0.99),
                comment=args.comment,
                save_path=save_folder,
                device="cuda" if torch.cuda.is_available() else "cpu",
                seed=int(training_params["seed"]),
            )
            
        # Manually set the eval_env since deepcopy doesn't work with complex environments
        model.eval_env = eval_env
        print("Evaluation environment successfully created and assigned.")
        
        # Train
        start_time = time.time()
        model.learn(int(training_params["learning_step"]))
        model.save()
        training_params["time"] = time.time() - start_time
        savers.save_as_csv(save_folder + "training_params_bptt.csv", training_params)
        
    else:
        # Testing mode
        test_model_path = save_folder + args.weight
        
        from test import Test
        env = LandingEnv(
            num_agent_per_scene=1,
            visual=True,
            random_kwargs=random_kwargs,
            scene_kwargs={
                "path": scene_path,
                "render_settings": {
                    "mode": "fix",
                    "view": "near",
                    "resolution": [1080, 1920],
                    "position": torch.tensor([[2., 0, 0]]),
                    "trajectory": True,
                    "line_width": 6.
                }
            },
            max_episode_steps=int(training_params["max_episode_steps"]),
        )
        env.reset()
        
        # Initialize BPTT model for testing
        model = BPTT(
            env=env,
            policy="MultiInputPolicy",
            policy_kwargs=dict(
                features_extractor_class=extractors.StateTargetImageExtractor,
                features_extractor_kwargs={
                    "net_arch": {
                        "color": {
                            "layer": [128],
                        },
                        "state": {
                            "layer": [128, 64],
                        },
                        "target": {
                            "layer": [128, 64],
                        }
                    }
                },
                net_arch=dict(
                    pi=[64, 64],
                    vf=[64, 64]),
                activation_fn=torch.nn.ReLU,
                optimizer_kwargs=dict(weight_decay=1e-5)
            ),
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
            model=wrapped,
            name=args.weight,
            save_path=os.path.dirname(os.path.realpath(__file__)) + "/saved/test"
        )
        test_handle.test(
            is_fig=True,
            is_fig_save=True,
            is_render=True,
            is_video=True,
            is_video_save=True,
            render_kwargs={}
        )

if __name__ == "__main__":
    main()
