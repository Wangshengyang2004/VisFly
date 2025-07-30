#!/usr/bin/env python3

import sys
import os
import time
import torch
import torch as th
import numpy as np
import traceback

# Add correct path for VisFly imports
sys.path.append(os.getcwd())

from VisFly.utils.launcher import rl_parser, training_params
args = rl_parser().parse_args()

from VisFly.utils.policies import extractors
from VisFly.utils.algorithms.BPTT import BPTT
from VisFly.utils import savers
from VisFly.envs.FlipEnv import FlipEnv
from VisFly.utils.type import Uniform

# GPU configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Enable gradient anomaly detection for debugging (disable for performance)
torch.autograd.set_detect_anomaly(True)  # Set to True for debugging

""" FLIP-SPECIFIC HYPERPARAMETERS """
# Number of parallel environments (agents) - FlipEnv benefits from more parallel experience
training_params["num_env"] = 256
# Total learning steps
training_params["learning_step"] = 2e7  # Reduced from navigation due to simpler task
# Comments and seed
training_params["comment"] = args.comment
training_params["seed"] = args.seed
# Episode length - shorter episodes for flip maneuver
training_params["max_episode_steps"] = 512  # Reduced from 256
# Learning rate for BPTT - higher for flip due to clearer reward signal
training_params["learning_rate"] = 1e-3
# BPTT horizon - shorter for flip dynamics
training_params["horizon"] = 128  # Reduced from 96
# Logging frequency
training_params["dump_step"] = 1000

# Directory where to save checkpoints and logs
save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + "/saved/"

# Random initialization optimized for flip training
random_kwargs = {
    "state_generator": {
        "class": "Uniform",
        "kwargs": [
            {
                # Start position: centered with small variations
                "position": {"mean": [0.0, 0.0, 1.5], "half": [0.3, 0.3, 0.2]},
                # Initial orientation: small perturbations from upright
                "orientation": {"mean": [0., 0., 0.], "half": [0.1, 0.1, 0.2]},
                # Initial velocity: near zero to start stable
                "velocity": {"mean": [0., 0., 0.], "half": [0.1, 0.1, 0.1]},
                # Initial angular velocity: small random spin to encourage flip initiation
                "angular_velocity": {"mean": [0., 0., 0.], "half": [0.5, 0.3, 0.3]},
            }
        ]
    }
}

# Dynamics configuration optimized for flip maneuvers
dynamics_kwargs = {
    "dt": 0.02,           # Higher frequency for flip precision
    "ctrl_dt": 0.02,      # Match simulation timestep
    "action_type": "bodyrate",  # Direct angular velocity control for flips
    "ctrl_delay": False,   # Disable delay for cleaner flip dynamics
    "cfg": "drone/drone_d435i",
    # Additional drag randomization for more realistic flip recovery
    "drag_random": 0.05,
}

# Scene configuration - simple scene for flip training with render settings
scene_kwargs = {
    "path": "datasets/visfly-beta/configs/scenes/garage_empty",
    "render_settings": {
        "mode": "fix",
        "view": "custom",
        "resolution": [1080, 1920],
        "position": th.tensor([[3., 3., 3.], [0., 0., 0.]]),  # Camera position and orientation
        "line_width": 8.,
        "trajectory": True,
        "axes": True,
    }
}

def main():
    print("FlipEnv BPTT Training")
    print("=" * 50)
    
    # Check if train flag is provided, default to training mode
    if not hasattr(args, 'train'):
        print("No arguments provided. Use --train to start training or --help for options.")
        return
    
    # Training mode
    if args.train:
        print(f"Initializing FlipEnv with {training_params['num_env']} parallel agents...")
        
        env = FlipEnv(
            num_agent_per_scene=int(training_params["num_env"]),
            random_kwargs=random_kwargs,
            dynamics_kwargs=dynamics_kwargs,
            scene_kwargs=scene_kwargs,
            visual=False,  # Disable visual for faster training
            requires_grad=True,  # Enable BPTT
            max_episode_steps=int(training_params["max_episode_steps"]),
            device="cpu",  # Environment on CPU, model on GPU
        )
        
        env.reset()
        print(f"✓ Environment initialized")
        print(f"  Agents: {env.num_envs}")
        print(f"  Max episode steps: {training_params['max_episode_steps']}")
        print(f"  BPTT horizon: {training_params['horizon']}")
        
        # Load pretrained model if provided
        if args.weight is not None:
            print(f"Loading pretrained model: {args.weight}")
            # Create model first, then load weights
            model = BPTT(
                env=env,
                policy="MultiInputPolicy",
                policy_kwargs=dict(
                    features_extractor_class=extractors.FlexibleExtractor,
                    features_extractor_kwargs=dict(
                        net_arch=dict(
                            state=dict(layer=[128, 64]),  # FlipEnv uses state-only observations
                        ),
                        activation_fn=torch.nn.ReLU,
                    ),
                    net_arch=dict(pi=[128, 64], qf=[128, 64]),  # Larger networks for complex flip dynamics
                    activation_fn=torch.nn.ReLU,
                    optimizer_kwargs=dict(weight_decay=1e-5),
                ),
                learning_rate=training_params["learning_rate"],
                comment=args.comment,
                save_path=save_folder,
                horizon=int(training_params["horizon"]),
                gamma=0.98,  # Slightly lower gamma for shorter episodes
                device="cuda" if torch.cuda.is_available() else "cpu",
                seed=int(training_params["seed"]),
                dump_step=int(training_params.get("dump_step", 100)),
            )
            model.load(os.path.join(save_folder, args.weight))
        else:
            print("Creating new BPTT model...")
            # Instantiate BPTT algorithm
            model = BPTT(
                env=env,
                policy="MultiInputPolicy",
                policy_kwargs=dict(
                    features_extractor_class=extractors.FlexibleExtractor,
                    features_extractor_kwargs=dict(
                        net_arch=dict(
                            state=dict(layer=[128, 64]),  # FlipEnv state processing
                        ),
                        activation_fn=torch.nn.ReLU,
                    ),
                    net_arch=dict(pi=[128, 64], qf=[128, 64]),  # Larger for flip complexity
                    activation_fn=torch.nn.ReLU,
                    optimizer_kwargs=dict(weight_decay=1e-5),
                ),
                learning_rate=training_params["learning_rate"],
                comment=args.comment,
                save_path=save_folder,
                horizon=int(training_params["horizon"]),
                gamma=0.98,  # Lower gamma for episodic flip task
                device="cuda" if torch.cuda.is_available() else "cpu",
                seed=int(training_params["seed"]),
                dump_step=int(training_params.get("dump_step", 100)),
            )
        
        print(f"✓ BPTT model created")
        print(f"  Learning rate: {training_params['learning_rate']}")
        print(f"  Device: {model.device}")
        print(f"  Gamma: 0.98")
        
        # Training loop with enhanced monitoring
        print("\\nStarting BPTT training...")
        print("Monitoring flip-specific metrics:")
        print("  - Flip progress completion rate")
        print("  - Stabilization success rate") 
        print("  - Average episode reward")
        print("  - Gradient norms")
        
        try:
            start_time = time.time()
            
            # Custom training loop with flip-specific logging
            model.learn(
                total_timesteps=int(training_params["learning_step"])
            )
            
            training_time = time.time() - start_time
            training_params["time"] = training_time
            
            print(f"\\n✓ Training completed in {training_time/3600:.2f} hours")
            
            # Save final model
            model.save()
            print(f"✓ Model saved to {save_folder}")
            
            # Save training parameters
            savers.save_as_csv(save_folder + "training_params.csv", training_params)
            print(f"✓ Training parameters saved")
            
        except KeyboardInterrupt:
            print("\\n⚠️ Training interrupted by user")
            print("Saving current model state...")
            model.save()
            training_params["time"] = time.time() - start_time
            training_params["status"] = "interrupted"
            savers.save_as_csv(save_folder + "training_params.csv", training_params)
            
        except Exception as e:
            print(f"\\n❌ Training failed with error: {e}")
            traceback.print_exc()
            
            # Save debug information
            training_params["error"] = str(e)
            training_params["time"] = time.time() - start_time
            savers.save_as_csv(save_folder + "training_params.csv", training_params)
            
    else:
        # Evaluation mode
        print("Evaluation mode")
        
        if args.weight is None:
            print("❌ No model weight specified for evaluation")
            print("Usage: python bptt.py --weight model_name.zip")
            return
            
        test_model_path = os.path.join(save_folder, args.weight)
        
        if not os.path.exists(test_model_path):
            print(f"❌ Model file not found: {test_model_path}")
            return
            
        print(f"Loading model: {args.weight}")
        
        # Create evaluation environment with visual rendering
        env = FlipEnv(
            num_agent_per_scene=1,
            random_kwargs=random_kwargs,
            dynamics_kwargs=dynamics_kwargs,
            scene_kwargs=scene_kwargs,
            visual=True,  # Enable visual for evaluation
            requires_grad=False,  # Disable BPTT for evaluation
            max_episode_steps=int(training_params["max_episode_steps"]),
            device="cpu",
        )
        
        env.reset()
        
        # Initialize BPTT model for testing
        model = BPTT(
            env=env,
            policy="MultiInputPolicy",
            policy_kwargs=dict(
                features_extractor_class=extractors.StateExtractor,
                features_extractor_kwargs=dict(
                    net_arch=dict(
                        state=dict(layer=[128, 64]),
                    ),
                    activation_fn=th.nn.ReLU,
                ),
                net_arch=dict(pi=[128, 64], qf=[128, 64]),
                activation_fn=th.nn.ReLU,
            ),
            learning_rate=float(training_params.get("learning_rate", 2e-3)),
            batch_size=int(training_params.get("batch_size", 200000)),
            buffer_size=int(training_params.get("buffer_size", 1000000)),
            horizon=int(training_params.get("horizon", 64)),
            device="cpu",
            seed=42,
        )
        
        # Load model weights
        model.load(test_model_path)
        
        # Import and run comprehensive test
        try:
            import traceback
            from test import Test
            
            # Initialize test handler
            test_handle = Test(
                env=env,
                model=model,
                name=args.weight,
                save_path=os.path.dirname(os.path.realpath(__file__)) + "/saved/test",
            )
            
            print("Running comprehensive flip evaluation...")
            
            # Run multiple episodes for robust evaluation
            num_episodes = 5  # Fewer episodes than navigation due to simpler task
            all_success_rates = []
            all_flip_completions = []
            all_episode_rewards = []
            
            for ep_i in range(num_episodes):
                print(f"\n=== Episode {ep_i + 1}/{num_episodes} ===")
                
                # Create episode directory
                episode_dir = os.path.join(test_handle.save_path, f"episode_{ep_i:03d}")
                if not os.path.exists(episode_dir):
                    os.makedirs(episode_dir, exist_ok=True)

                # Print agent start information for this episode
                print(f"Agent start position: {env.position[0].cpu().numpy()}")
                print(f"Agent start orientation: {env.orientation[0].cpu().numpy()}")
                
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
                    # Run episode with comprehensive analysis
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
                    
                    # Generate flip-specific debug analysis plots
                    try:
                        debug_figs = test_handle.draw_debug()
                        print(f"Episode {ep_i}: Debug analysis completed")
                    except Exception as e:
                        print(f"Error during draw_debug: {e}")
                        traceback.print_exc()
                        
                except Exception as e:
                    print(f"Error during test execution: {e}")
                    traceback.print_exc()
                    print("Continuing with next episode...")
                    continue

                # Save all figures manually into the episode folder with meaningful names
                if figs:
                    for fig_idx, fig in enumerate(figs):
                        fig_path = os.path.join(episode_dir, f"trajectory_plot_{fig_idx}.png")
                        try:
                            fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                            print(f"Episode {ep_i}: trajectory plot saved in {fig_path}")
                        except Exception as e:
                            print(f"Error saving figure {fig_idx}: {e}")
                else:
                    print(f"No figures to save for episode {ep_i}")

                # Save videos (combined, global, and individual agent views)
                try:
                    test_handle.save_combined_video(episode_dir)
                    print(f"Episode {ep_i}: Videos saved to {episode_dir}")
                except Exception as e:
                    print(f"Error saving videos for episode {ep_i}: {e}")
                    traceback.print_exc()

                # Calculate episode statistics
                if hasattr(test_handle, 'reward_all') and test_handle.reward_all:
                    episode_reward = sum([r.sum() if hasattr(r, 'sum') else r for r in test_handle.reward_all])
                    all_episode_rewards.append(float(episode_reward))
                    print(f"Episode {ep_i}: Total reward = {episode_reward:.2f}")

                # Calculate flip-specific metrics
                if hasattr(test_handle, 'info_all') and test_handle.info_all:
                    success_count = 0
                    flip_completion_count = 0
                    total_steps = len(test_handle.info_all)
                    
                    for timestep_info in test_handle.info_all:
                        for agent_info in timestep_info:
                            if agent_info:
                                if agent_info.get("is_success", False):
                                    success_count += 1
                                # Check flip completion based on reward components or info
                                if "flip_progress" in agent_info and agent_info["flip_progress"] > 0.95:
                                    flip_completion_count += 1
                    
                    success_rate = success_count / max(total_steps, 1)
                    flip_completion_rate = flip_completion_count / max(total_steps, 1)
                    
                    all_success_rates.append(success_rate)
                    all_flip_completions.append(flip_completion_rate)
                    
                    print(f"Episode {ep_i}: Success rate = {success_rate:.2%}")
                    print(f"Episode {ep_i}: Flip completion rate = {flip_completion_rate:.2%}")
                    print(f"Episode {ep_i}: Episode length = {total_steps} steps")

            # Print comprehensive evaluation summary
            print("\n" + "="*60)
            print("FLIP EVALUATION SUMMARY")
            print("="*60)
            print(f"Episodes completed: {len(all_episode_rewards)}")
            
            if all_episode_rewards:
                print(f"Average episode reward: {sum(all_episode_rewards)/len(all_episode_rewards):.2f}")
                print(f"Best episode reward: {max(all_episode_rewards):.2f}")
                print(f"Worst episode reward: {min(all_episode_rewards):.2f}")
            
            if all_success_rates:
                avg_success = sum(all_success_rates) / len(all_success_rates)
                print(f"Average success rate: {avg_success:.2%}")
                
            if all_flip_completions:
                avg_flip = sum(all_flip_completions) / len(all_flip_completions)
                print(f"Average flip completion rate: {avg_flip:.2%}")
                
            print(f"Results saved to: {test_handle.save_path}")
            print("="*60)
            
            print("✓ Comprehensive flip evaluation completed!")
            
        except ImportError:
            print("⚠️ Test module not found, running basic evaluation...")
            
            # Basic evaluation loop
            obs = env.reset()
            total_reward = 0
            flip_completions = 0
            stabilizations = 0
            
            for step in range(training_params["max_episode_steps"]):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                
                # Monitor flip progress
                if hasattr(env, 'flip_progress'):
                    if env.flip_progress[0] > 0.95:
                        flip_completions += 1
                if hasattr(env, 'stabilization_progress'):
                    if env.stabilization_progress[0] > 0.8:
                        stabilizations += 1
                        
                if done:
                    break
            
            print(f"Evaluation Results:")
            print(f"  Total reward: {total_reward:.2f}")
            print(f"  Flip completion steps: {flip_completions}")
            print(f"  Stabilization steps: {stabilizations}")
            print(f"  Episode length: {step + 1}")

if __name__ == "__main__":
    main()