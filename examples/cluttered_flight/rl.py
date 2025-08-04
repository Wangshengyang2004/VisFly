#!/usr/bin/env python3

import sys
import os
import traceback
import numpy as np
import torch
import time

sys.path.append(os.getcwd())
from VisFly.utils.policies import extractors
from VisFly.utils.algorithms.ppo import ppo
from VisFly.utils import savers
import torch as th
from VisFly.envs.NavigationEnv_old import NavigationEnv
from VisFly.utils.launcher import rl_parser, training_params
from VisFly.utils.type import Uniform
from habitat_sim.sensor import SensorType
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
args = rl_parser().parse_args()
""" SAVED HYPERPARAMETERS """
training_params["num_env"] = 48
training_params["learning_step"] = 1e7
training_params["comment"] = args.comment
training_params["max_episode_steps"] = 256
training_params["n_steps"] = training_params["max_episode_steps"]
training_params["batch_size"] = training_params["num_env"] * training_params["n_steps"]
training_params["learning_rate"] = 1e-3
save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + "/saved/"
random_kwargs = {
    "state_generator": {
        "class": "Uniform",
        "kwargs": [
            {
                "position": {"mean": [6., -2., 1.], "half": [.50, .50, .50]},
                # "orientation": {"mean": [0., 0., 0.], "half": [0.0, 0.0, 3.1416]},
                # "velocity": {"mean": [0., 0., 0.], "half": [0.1, 0.1, 0.1]},
                # "angular_velocity": {"mean": [0., 0., 0.], "half": [1., 1., 1.]},
            }
        ]
    }
}

scene_kwargs = {
    "path": "datasets/visfly-beta/configs/scenes/box15_wall_box15_wall"
}

dynamics_kwargs = {
    "dt": 0.03,
    "ctrl_dt": 0.03,
    "action_type": "bodyrate",
    "ctrl_delay": True,
    "cfg": "drone/drone_d435i",
}

def main():
    # if train mode, train the model
    if args.train:
        # add depth sensor for training
        sensor_kwargs = [{
            "sensor_type": SensorType.DEPTH,
            "uuid": "depth",
            "resolution": [64, 64],
        }]
        env = NavigationEnv(num_agent_per_scene=training_params["num_env"],
                            random_kwargs=random_kwargs,
                            sensor_kwargs=sensor_kwargs,
                            visual=True,
                            max_episode_steps=training_params["max_episode_steps"],
                            scene_kwargs=scene_kwargs,
                            dynamics_kwargs=dynamics_kwargs,
                            target=th.tensor([6., -8., 1.]),
                            device="cpu",
                            )

        if args.weight is not None:
            model = ppo.load(save_folder + args.weight, env=env)
        else:
            model = ppo(
                policy="CustomMultiInputPolicy",
                policy_kwargs=dict(
                    features_extractor_class=extractors.StateTargetImageExtractor,
                    features_extractor_kwargs={
                        "net_arch": {
                            "depth": {
                                # "backbone": "mobilenet_s",
                                "layer": [128],
                            },
                            "state": {
                                "layer": [128, 64],
                            },
                            "target": {
                                "layer": [128, 64],
                            },
                            # "recurrent":{
                            #     "class": "GRU",
                            #     "kwargs":{
                            #         "hidden_size": latent_dim,
                            #     }
                            # }
                        }
                    },
                    net_arch={
                        "pi": [64, 64],
                        "vf": [64, 64],

                    },
                    activation_fn=torch.nn.ReLU,
                    optimizer_kwargs=dict(weight_decay=1e-5),
                ),
                env=env,
                verbose=training_params["verbose"],
                tensorboard_log=save_folder,
                gamma=training_params["gamma"],  # lower 0.9 ~ 0.99
                n_steps=training_params["n_steps"],
                ent_coef=training_params["ent_coef"],
                learning_rate=training_params["learning_rate"],
                vf_coef=training_params["vf_coef"],
                max_grad_norm=training_params["max_grad_norm"],
                batch_size=training_params["batch_size"],
                gae_lambda=training_params["gae_lambda"],
                n_epochs=training_params["n_epochs"],
                clip_range=training_params["clip_range"],
                device="cuda",
                seed=training_params["seed"],
                comment=args.comment,
            )

        start_time = time.time()
        model.learn(training_params["learning_step"])
        model.save()
        training_params["time"] = time.time() - start_time

        savers.save_as_csv(save_folder + "training_params.csv", training_params)

    # Testing mode with a trained weight
    else:
        test_model_path = save_folder + args.weight
        test_scene_kwargs = {
            "path": "VisFly/datasets/visfly-beta/configs/scenes/box15_wall_box15_wall",
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
        sensor_kwargs = [{
            "sensor_type": SensorType.DEPTH,
            "uuid": "depth",
            "resolution": [64, 64],
        }]
        sensor_kwargs = [{
            "sensor_type": SensorType.DEPTH,
            "uuid": "depth",
            "resolution": [64, 64],
        }]
        env = NavigationEnv(num_agent_per_scene=1, visual=True,
                            random_kwargs=random_kwargs,
                            scene_kwargs=test_scene_kwargs,
                            sensor_kwargs=sensor_kwargs,
                            dynamics_kwargs=dynamics_kwargs,
                            target=th.tensor([6., -8., 1.]),
                            device="cpu",
                            )

        model = ppo.load(test_model_path, env=env)

        class ModelWrapper:
            def __init__(self, policy):
                self.policy = policy
                self.env = env
        wrapped = ModelWrapper(model.policy)
        test_handle = Test(
            env,  # First parameter: env
            wrapped,  # Second parameter: model (with .policy attribute)
            "rl_test",  # Third parameter: name (as string)
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

            # Print agent start position for this episode
            start_position = env.position[0].cpu().numpy()
            target_position = env.target[0].cpu().numpy()
            start_distance = ((env.target[0] - env.position[0]).norm()).item()
            print(f"\n=== Episode {ep_i} ===")
            print(f"Agent start position: {start_position}")
            print(f"Target position: {target_position}")
            print(f"Start distance to target: {start_distance:.2f}m")

            # Clear accumulated data from previous episodes
            test_handle.obs_all = []
            test_handle.state_all = []
            test_handle.info_all = []
            test_handle.action_all = []
            test_handle.collision_all = []
            test_handle.render_image_all = []
            test_handle.reward_all = []
            test_handle.reward_components = []
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

            # Print agent end position for this episode
            end_position = env.position[0].cpu().numpy()
            end_distance = ((env.target[0] - env.position[0]).norm()).item()
            print(f"Agent end position: {end_position}")
            print(f"End distance to target: {end_distance:.2f}m")
            print(f"Distance traveled: {((th.tensor(end_position) - th.tensor(start_position)).norm()).item():.2f}m")

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