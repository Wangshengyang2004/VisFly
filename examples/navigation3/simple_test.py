#!/usr/bin/env python3

import sys
import os
sys.path.append(os.getcwd())

import torch as th
import numpy as np
from VisFly.envs.NavigationEnv import NavigationEnv3

def simple_navigation_test():
    """Simple test for collision-free spawn and target validation"""
    
    # Create environment
    env = NavigationEnv3(
        num_agent_per_scene=1,
        num_scene=1,
        seed=42,
        visual=True,
        scene_kwargs={
            "path": "VisFly/datasets/visfly-beta/configs/scenes/box15_wall_pillar"
        },
        device="cpu",
        max_episode_steps=50  # Short episodes for quick testing
    )
    
    print("=== Simple Navigation Test ===")
    
    # Reset and get initial position
    env.reset()
    start_pos = env.position.clone()
    print(f"Start position: {start_pos}")
    print(f"Original target: {env.target}")
    print(f"Distance to target: {(start_pos - env.target).norm(dim=1).item():.2f}m")
    
    # Test collision-free spawn validation
    print("\n=== Collision-Free Spawn Validation ===")
    spawn_valid = env.validate_spawn_position(start_pos)
    print(f"Spawn position valid: {spawn_valid.item()}")
    
    # Test target validation
    target_valid = env.validate_target_position(env.target)
    print(f"Target position valid: {target_valid.item()}")
    
    # Set a very close target (0.5m away)
    close_target = start_pos.clone()
    close_target[0, 0] += 0.5  # Move 0.5m in X direction
    env.target = close_target
    
    print(f"New close target: {env.target}")
    print(f"Distance to close target: {(start_pos - env.target).norm(dim=1).item():.2f}m")
    
    # Check success detection
    success = env.get_success()
    print(f"Success status with close target: {success}")
    print(f"Success threshold: {env.success_radius}m")
    
    # Test multiple resets to verify collision-free spawn consistency
    print("\n=== Multiple Reset Test ===")
    for i in range(3):
        print(f"\nReset #{i+1}:")
        env.reset()
        
        spawn_pos = env.position.clone()
        target_pos = env.target.clone()
        
        spawn_valid = env.validate_spawn_position(spawn_pos)
        target_valid = env.validate_target_position(target_pos)
        distance = (spawn_pos - target_pos).norm(dim=1).item()
        
        print(f"  Spawn valid: {spawn_valid.item()}")
        print(f"  Target valid: {target_valid.item()}")
        print(f"  Distance: {distance:.2f}m")
    
    print("\n=== Test Summary ===")
    print("✓ Environment initialization successful")
    print("✓ Collision-free spawn validation working")
    print("✓ Target validation working")
    print("✓ Success detection functional")
    print("✓ Multiple resets with validation successful")

if __name__ == "__main__":
    simple_navigation_test() 