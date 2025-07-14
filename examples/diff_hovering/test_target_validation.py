#!/usr/bin/env python3

import sys
import os
sys.path.append(os.getcwd())

import torch as th
import numpy as np
from VisFly.envs.HoverEnv import HoverEnv2

def test_target_validation():
    """Test the target validation functionality"""
    
    # Create environment
    env = HoverEnv2(
        num_agent_per_scene=1,
        num_scene=1,
        seed=42,
        visual=True,
        scene_kwargs={
            "path": "VisFly/datasets/visfly-beta/configs/scenes/garage_empty",
            "render_settings": {
                "mode": "fix",
                "view": "custom",
                "resolution": [1080, 1920],
                "position": th.tensor([[7., 6.8, 5.5], [7, 4.8, 4.5]]),
                "trajectory": True,
            }
        },
        device="cpu"
    )
    
    # Reset environment to get initial state
    env.reset()
    
    print("Testing target validation functionality...")
    print(f"Current agent position: {env.position}")
    print(f"Current target: {env.target}")
    
    # Test 1: Validate current target
    print("\n1. Testing validation of current target:")
    is_valid = env.validate_target_position(env.target)
    print(f"Current target is valid: {is_valid}")
    
    # Test 2: Generate valid targets
    print("\n2. Testing generation of valid targets:")
    start_positions = env.position
    valid_targets = env.generate_valid_targets(
        start_positions=start_positions,
        num_targets=env.num_envs,
        min_distance=2.0,
        max_distance=8.0,
        max_attempts=50
    )
    print(f"Generated valid targets: {valid_targets}")
    
    # Test 3: Validate generated targets
    print("\n3. Testing validation of generated targets:")
    validation_result = env.validate_target_position(valid_targets)
    print(f"Generated targets are valid: {validation_result}")
    
    # Test 4: Set valid targets
    print("\n4. Testing setting of valid targets:")
    success = env.set_valid_targets(valid_targets)
    print(f"Successfully set valid targets: {success}")
    print(f"New target: {env.target}")
    
    # Test 5: Test invalid targets (inside obstacles)
    print("\n5. Testing invalid targets (inside obstacles):")
    # Create some potentially invalid targets (very close to obstacles)
    invalid_targets = th.tensor([
        [0.0, 0.0, 0.5],  # Very low Z, might be inside ground
        [10.0, 10.0, 1.5],  # Far outside scene bounds
        [1.0, 0.0, 1.5],  # This should be valid
    ], device=env.device)
    
    validation_result = env.validate_target_position(invalid_targets)
    print(f"Invalid targets validation: {validation_result}")
    
    # Test 6: Try to set invalid targets
    print("\n6. Testing setting of invalid targets:")
    success = env.set_valid_targets(invalid_targets)
    print(f"Successfully set invalid targets: {success}")
    
    print("\nTarget validation test completed!")

if __name__ == "__main__":
    test_target_validation() 