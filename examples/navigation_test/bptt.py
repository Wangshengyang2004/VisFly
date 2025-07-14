#!/usr/bin/env python3

import sys
import os
import time
import torch
import torch as th
import numpy as np
import traceback

sys.path.append(os.getcwd())

from VisFly.utils.launcher import rl_parser, training_params
args = rl_parser().parse_args()

from VisFly.utils.policies import extractors
from VisFly.utils.algorithms.BPTT import BPTT
from VisFly.utils import savers
from VisFly.envs.NavigationEnv import NavigationEnv2
from VisFly.utils.type import Uniform
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Disable gradient anomaly detection for physics-based BPTT
# torch.autograd.set_detect_anomaly(True)  # Disabled for stability

""" SAVED HYPERPARAMETERS """
# Number of parallel environments (agents)
training_params["num_env"] = 150
# Total learning steps
training_params["learning_step"] = 1e7
# Comments and seed
training_params["comment"] = args.comment
training_params["seed"] = args.seed
# Episode length
training_params["max_episode_steps"] = 256
# Learning rate for BPTT
training_params["learning_rate"] = 1e-3
# BPTT horizon
training_params["horizon"] = 96
# Logging frequency
training_params["dump_step"] = 50

# Directory where to save checkpoints and logs
save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + "/saved/"

# Random initialization for environment resets
random_kwargs = {
    "state_generator": {
        "class": "Uniform",
        "kwargs": [
            {
                "position": {"mean": [3., 0., 1.], "half": [0.5, 0.5, 0.5]},
                # "orientation": {"mean": [0., 0., 0.], "half": [0.0, 0.0, 3.1416]},
                # "velocity": {"mean": [0., 0., 0.], "half": [0.1, 0.1, 0.1]},
                # "angular_velocity": {"mean": [0., 0., 0.], "half": [1., 1., 1.]},
            }
        ]
    }
}

# Scene configuration for visual rendering
scene_kwargs = {
    "path": "VisFly/datasets/visfly-beta/configs/scenes/box15_wall_pillar"
}

# Dynamics configuration 
dynamics_kwargs = {
    "dt": 0.03,
    "ctrl_dt": 0.03,
    "action_type": "bodyrate",
    "ctrl_delay": True,
    "cfg": "drone/drone_d435i",
}

def main():
    # Training mode
    if args.train:
        env = NavigationEnv2(
            num_agent_per_scene=int(training_params["num_env"]),
            random_kwargs=random_kwargs,
            dynamics_kwargs=dynamics_kwargs,
            scene_kwargs=scene_kwargs,
            visual=True, 
            requires_grad=True,
            max_episode_steps=int(training_params["max_episode_steps"]),
            target=th.tensor([[10., 0., 1.]]),
            device="cpu",
            tensor_output=True,
        )
        
        env.reset()
        
        # Enable individual reward component logging during evaluation only
        # Training requires tensor returns, evaluation will use dict returns
        setattr(env, '_enable_individual_rewards', False)
        
        # Load pretrained model if provided
        if args.weight is not None:
            # Create model first, then load weights
            model = BPTT(
                env=env,
                policy="MultiInputPolicy",
                policy_kwargs=dict(
                    features_extractor_class=extractors.StateTargetExtractor,
                    features_extractor_kwargs=dict(
                        net_arch=dict(
                            # depth=dict(layer=[128]),
                            state=dict(layer=[128, 64]),
                            target=dict(layer=[128, 64]),
                        ),
                        activation_fn=torch.nn.ReLU,
                    ),
                    net_arch=dict(pi=[64, 64], qf=[64, 64]),
                    activation_fn=torch.nn.ReLU,
                    optimizer_kwargs=dict(weight_decay=1e-5),
                ),
                learning_rate=training_params["learning_rate"],
                comment=args.comment,
                save_path=save_folder,
                horizon=int(training_params["horizon"]),
                gamma=training_params.get("gamma", 0.99),
                device="cuda",
                seed=int(training_params["seed"]),
                dump_step=int(training_params.get("dump_step", 1000)),
            )
            model.load(os.path.join(save_folder + args.weight))
        else:
            # Instantiate BPTT algorithm
            model = BPTT(
                env=env,
                policy="MultiInputPolicy",
                policy_kwargs=dict(
                    features_extractor_class=extractors.StateTargetExtractor,
                    features_extractor_kwargs=dict(
                        net_arch=dict(
                            depth=dict(layer=[128]),
                            state=dict(layer=[128, 64]),
                            target=dict(layer=[128, 64]),
                        ),
                        activation_fn=torch.nn.ReLU,
                    ),
                    net_arch=dict(pi=[64, 64], qf=[64, 64]),
                    activation_fn=torch.nn.ReLU,
                    optimizer_kwargs=dict(weight_decay=1e-5),
                ),
                learning_rate=training_params["learning_rate"],
                comment=args.comment,
                save_path=save_folder,
                horizon=int(training_params["horizon"]),
                gamma=training_params.get("gamma", 0.99),
                device="cuda",
                seed=int(training_params["seed"]),
                dump_step=int(training_params.get("dump_step", 1000)),
            )
        
        # Train
        start_time = time.time()
        model.learn(int(training_params["learning_step"]))
        model.save()
        training_params["time"] = time.time() - start_time
        savers.save_as_csv(save_folder + "training_params_bptt.csv", training_params)
    else:
        # Testing mode
        test_model_path = save_folder + args.weight
        
        # Add render settings for test environment
        test_scene_kwargs = {
            "path": "VisFly/datasets/visfly-beta/configs/scenes/box15_wall_pillar",
            "render_settings": {
                "mode": "fix",
                "view": "custom",
                "resolution": [1080, 1920],
                "position": torch.tensor([[14., 0., 6.], [0., -2., 0.]]),  # Angled position for good coverage
                "line_width": 8.,
                "trajectory": True,
                "axes": True,
            }
        }
        
        from tst import Test
        env = NavigationEnv2(
            num_agent_per_scene=1,
            random_kwargs=random_kwargs,
            dynamics_kwargs=dynamics_kwargs,
            scene_kwargs=test_scene_kwargs,  # Use test scene kwargs with render settings
            visual=True,  # Enable visual for depth sensor
            max_episode_steps=int(training_params["max_episode_steps"]),
            target=th.tensor([[10., 0., 1.]]),
            device="cpu",
            tensor_output=True,
        )
        env.reset()
        
        # Initialize BPTT model for testing
        model = BPTT(
            env=env,
            policy="MultiInputPolicy",
            policy_kwargs=dict(
                features_extractor_class=extractors.StateTargetExtractor,
                features_extractor_kwargs=dict(
                    net_arch=dict(
                        # depth=dict(layer=[128]),
                        state=dict(layer=[128, 64]),
                        target=dict(layer=[128, 64]),
                    ),
                    activation_fn=torch.nn.ReLU,
                ),
                net_arch=dict(pi=[64, 64], qf=[64, 64]),
                activation_fn=torch.nn.ReLU,
                optimizer_kwargs=dict(weight_decay=1e-5),
            ),
            device="cpu"
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
            env,  # First parameter: env
            wrapped,  # Second parameter: model (with .policy attribute)
            "bptt_test",  # Third parameter: name (as string)
            os.path.dirname(os.path.realpath(__file__)) + "/saved/test",  # Fourth parameter: save_path
        )
        # --- Extended evaluation -------------------------------------------------
        n_eval_episodes = 4  # how many complete episodes to evaluate for SR alignment
        aggregated_success = 0
        total_agents_eval = env.num_envs * n_eval_episodes

        for ep_i in range(n_eval_episodes):
            # create a sub-directory for this episode:  .../bptt_test/episode_XX/
            episode_dir = os.path.join(test_handle.save_path, f"episode_{ep_i:03d}")
            if not os.path.exists(episode_dir):
                os.makedirs(episode_dir, exist_ok=True)

            # Clear accumulated data from previous episodes
            test_handle.obs_all = []
            test_handle.state_all = []
            test_handle.info_all = []
            test_handle.action_all = []
            test_handle.collision_all = []
            test_handle.render_image_all = []
            test_handle.reward_all = []
            test_handle.t = []
            test_handle.eq_r = []
            test_handle.eq_l = []

            try:
                # Skip interactive video playback, only save videos to files
                result = test_handle.test(
                    is_fig=True,              # draw trajectory plots every episode
                    is_fig_save=False,        # disable auto-save to avoid conflicts
                    is_video=False,           # disable interactive video playback
                    is_video_save=True,       # enable video saving to files
                    is_sub_video=False,       # disable sub video
                )
                # Unpack result robustly
                if isinstance(result, tuple):
                    figs = result[0] if len(result) > 0 else []
                else:
                    figs = result
                # Ensure figs is always a list
                import matplotlib.figure
                if figs is not None and not isinstance(figs, list):
                    if isinstance(figs, matplotlib.figure.Figure):
                        figs = [figs]
                    else:
                        figs = list(figs)
                # Generate debug analysis plots
                try:
                    _debug_figs = test_handle.draw_debug()  # Use _ prefix to suppress unused variable warning
                except Exception as e:
                    print(f"Error during draw_debug: {e}")
                    traceback.print_exc()
            except Exception as e:
                print(f"Error during test execution: {e}")
                traceback.print_exc()
                print("Continuing with next episode...")
                continue

            # save all figures manually into the episode folder with meaningful names
            if figs:
                for fig_idx, fig in enumerate(figs):
                    fig_path = os.path.join(episode_dir, f"trajectory_plot_{fig_idx}.png")
                    try:
                        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                        print(f"Episode {ep_i}: fig saved in {fig_path}")
                    except Exception as e:
                        print(f"Error saving figure {fig_idx}: {e}")
            else:
                print(f"No figures to save for episode {ep_i}")

            # Save videos (combined, global, and individual agent views)
            try:
                test_handle.save_combined_video(episode_dir)
            except Exception as e:
                print(f"Error saving combined video for episode {ep_i}: {e}")
                print("Continuing with next episode...")

            # Count agents that achieved success at any point during the episode
            successful_agents = set()
            for timestep_idx, timestep_info in enumerate(test_handle.info_all):
                for agent_idx, agent_info in enumerate(timestep_info):
                    if agent_info and "is_success" in agent_info and agent_info["is_success"]:
                        successful_agents.add(agent_idx)
            episode_successes = len(successful_agents)
            aggregated_success += episode_successes
            print(f"Episode {ep_i}: {episode_successes}/{env.num_envs} agents reached target at some point")

        eval_sr = aggregated_success / total_agents_eval
        print(f"\nAggregated evaluation over {n_eval_episodes} episodes → Success Rate: {eval_sr:.3f}\n")

if __name__ == "__main__":
    main() 