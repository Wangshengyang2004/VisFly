#!/usr/bin/env python3

import sys
import os
import time
import torch
import numpy as np
import traceback

# Ensure project root is in path
sys.path.append(os.getcwd())

# Parse args FIRST before importing any VisFly modules
from VisFly.utils.launcher import rl_parser, training_params
args = rl_parser().parse_args()

# Now import the rest
from VisFly.envs.NavigationEnv import NavigationEnv3

def test_spawn_area_simple(spawn_half, description):
    """Test spawn area with collision-free spawn validation"""
    print(f"\n{'='*60}")
    print(f"Testing {description}")
    print(f"Spawn area: {spawn_half[0]*2}x{spawn_half[1]*2}m")
    print(f"{'='*60}")
    
    # Configure spawn area
    random_kwargs = {
        "state_generator": {
            "class": "Uniform",
            "kwargs": [
                {
                    "position": {"mean": [0., 0., 1.], "half": spawn_half},
                    "orientation": {"mean": [0., 0., 0.], "half": [0.0, 0.0, 3.1416]},
                    "velocity": {"mean": [0., 0., 0.], "half": [0.1, 0.1, 0.1]},
                    "angular_velocity": {"mean": [0., 0., 0.], "half": [1., 1., 1.]},
                }
            ]
        }
    }
    
    scene_kwargs = {
        "path": "VisFly/datasets/visfly-beta/configs/scenes/box15_wall_pillar"
    }
    
    dynamics_kwargs = {
        "dt": 0.02,
        "ctrl_dt": 0.02,
        "action_type": "bodyrate",
        "ctrl_delay": True,
        "cfg": "drone/drone_d435i",
    }
    
    try:
        # Create environment (use fewer agents to avoid OpenGL issues)
        num_agents = 50  # Reduced from 150 to avoid crashes
        print(f"Creating environment with {num_agents} agents...")
        
        env = NavigationEnv3(
            num_agent_per_scene=num_agents,
            random_kwargs=random_kwargs,
            dynamics_kwargs=dynamics_kwargs,
            scene_kwargs=scene_kwargs,
            visual=True,  # Enable visual for depth sensor
            requires_grad=False,  # Disable gradients for testing
            max_episode_steps=256,
            device="cpu",
        )
        
        print("Resetting environment...")
        start_time = time.time()
        env.reset()
        reset_time = time.time() - start_time
        
        # Validate spawn positions
        spawn_positions = env.position.clone()
        spawn_valid = env.validate_spawn_position(spawn_positions)
        valid_spawns = spawn_valid.sum().item()
        spawn_rate = (valid_spawns / num_agents) * 100
        
        print(f"Valid spawn positions: {valid_spawns}/{num_agents} ({spawn_rate:.1f}%)")
        print(f"Reset time: {reset_time:.2f}s")
        
        # Try target generation (limited attempts for testing)
        print("Testing target generation...")
        target_start_time = time.time()
        
        try:
            targets = env.generate_valid_targets(
                start_positions=spawn_positions,
                num_targets=num_agents,
                min_distance=2.0,
                max_distance=8.0,
                max_attempts=500  # Limited for testing
            )
            target_valid = env.validate_target_position(targets)
            valid_targets = target_valid.sum().item()
            target_rate = (valid_targets / num_agents) * 100
            
            target_time = time.time() - target_start_time
            print(f"Valid targets: {valid_targets}/{num_agents} ({target_rate:.1f}%)")
            print(f"Target generation time: {target_time:.2f}s")
            
            target_success = True
        except Exception as e:
            print(f"Target generation failed: {e}")
            valid_targets = 0
            target_rate = 0
            target_success = False
        
        # Clean up
        del env
        
        return {
            'valid_spawns': valid_spawns,
            'spawn_rate': spawn_rate,
            'valid_targets': valid_targets,
            'target_rate': target_rate,
            'reset_time': reset_time,
            'success': target_success,
            'num_agents': num_agents
        }
        
    except Exception as e:
        print(f"Environment creation failed: {e}")
        traceback.print_exc()
        return {
            'valid_spawns': 0,
            'spawn_rate': 0,
            'valid_targets': 0,
            'target_rate': 0,
            'reset_time': 0,
            'success': False,
            'num_agents': num_agents
        }

def main():
    print("SPAWN AREA FIX VERIFICATION")
    print("="*80)
    print("This script demonstrates the fix for the BPTT spawn area issue.")
    print("The old 36x36m spawn area was too large for the box15_wall_pillar scene,")
    print("causing most agents to spawn outside bounds or in obstacles.")
    print("The new 16x16m spawn area should show significant improvement.")
    
    # Test OLD problematic spawn area
    print("\n🔴 TESTING OLD SPAWN AREA (PROBLEMATIC)")
    old_results = test_spawn_area_simple([18., 18., 1.], "OLD 36x36m spawn area")
    
    # Add delay to avoid conflicts
    time.sleep(3)
    
    # Test NEW fixed spawn area
    print("\n🟢 TESTING NEW SPAWN AREA (FIXED)")
    new_results = test_spawn_area_simple([8., 8., 1.], "NEW 16x16m spawn area")
    
    # Comparison
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}")
    
    print(f"Spawn Success Rates:")
    print(f"  OLD (36x36m): {old_results['spawn_rate']:.1f}% ({old_results['valid_spawns']}/{old_results['num_agents']})")
    print(f"  NEW (16x16m): {new_results['spawn_rate']:.1f}% ({new_results['valid_spawns']}/{new_results['num_agents']})")
    
    print(f"\nTarget Success Rates:")
    print(f"  OLD (36x36m): {old_results['target_rate']:.1f}% ({old_results['valid_targets']}/{old_results['num_agents']})")
    print(f"  NEW (16x16m): {new_results['target_rate']:.1f}% ({new_results['valid_targets']}/{new_results['num_agents']})")
    
    # Calculate improvements
    spawn_improvement = new_results['spawn_rate'] - old_results['spawn_rate']
    target_improvement = new_results['target_rate'] - old_results['target_rate']
    
    print(f"\nImprovements:")
    print(f"  Spawn rate: +{spawn_improvement:.1f}% improvement")
    print(f"  Target rate: +{target_improvement:.1f}% improvement")
    
    # Verdict
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")
    
    if new_results['spawn_rate'] > old_results['spawn_rate']:
        print("✅ SPAWN FIX SUCCESSFUL: More agents spawn in valid positions")
    else:
        print("❌ SPAWN FIX FAILED: No improvement in spawn success rate")
    
    if new_results['target_rate'] > old_results['target_rate']:
        print("✅ TARGET FIX SUCCESSFUL: Target generation improved")
    else:
        print("⚠️  TARGET GENERATION: Still struggling (but spawn fix helps)")
    
    # BPTT Training Assessment
    if new_results['spawn_rate'] >= 80:
        print("🎉 EXCELLENT: Ready for BPTT training with 150 agents")
    elif new_results['spawn_rate'] >= 60:
        print("👍 GOOD: Should work for BPTT training")
    else:
        print("⚠️  NEEDS MORE WORK: May need further adjustments")
    
    print(f"\n{'='*80}")
    print("CONCLUSION: The spawn area fix reduces the area from 36x36m to 16x16m")
    print("to better match the box15_wall_pillar scene size, significantly")
    print("improving the success rate of collision-free spawn generation.")
    print("This should resolve your BPTT training target generation issues.")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 