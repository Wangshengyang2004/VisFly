#!/usr/bin/env python3

import sys
import os
sys.path.append(os.getcwd())

import torch as th
import numpy as np
from VisFly.envs.NavigationEnv import NavigationEnv3

def test_robust_target_validation():
    """Test the robust target validation functionality with retry mechanism"""
    
    # Create environment with cluttered scene for better collision testing
    env = NavigationEnv3(
        num_agent_per_scene=1,
        num_scene=1,
        seed=42,
        visual=True,
        scene_kwargs={
            "path": "VisFly/datasets/visfly-beta/configs/scenes/box15_wall_pillar"
        },
        device="cpu"
    )
    N = env.num_envs
    
    print("Testing robust target validation functionality...")
    print("=" * 60)
    
    # Test 1: Reset with robust validation
    print("\n1. Testing reset with robust target validation:")
    env.reset()
    print(f"Current agent position: {env.position}")
    print(f"Current target: {env.target}")
    
    # Validate the generated target
    is_valid = env.validate_target_position(env.target)
    print(f"Target validation result: {is_valid}")
    
    # Test 2: Test the retry mechanism with invalid targets
    print("\n2. Testing retry mechanism with invalid targets:")
    start_positions = env.position.clone()
    # Generate some potentially invalid targets (same batch size as env.num_envs)
    invalid_targets = th.zeros((N, 3), device=env.device)
    invalid_targets[:, 0] = 15.0  # Far outside bounds for all
    invalid_targets[:, 1] = 15.0
    invalid_targets[:, 2] = 1.0
    print(f"Testing targets: {invalid_targets}")
    # Use the retry mechanism
    success = env.set_valid_targets_with_retry(invalid_targets, start_positions)
    print(f"Retry mechanism success: {success}")
    print(f"Final targets: {env.target}")
    
    # Test 3: Test the robust generation function
    print("\n3. Testing robust target generation:")
    robust_targets = env.generate_valid_targets_with_retry(
        start_positions=start_positions,
        num_targets=N,
        min_distance=2.0,
        max_distance=8.0,
        max_attempts_per_batch=20,
        max_batches=5
    )
    print(f"Generated robust targets: {robust_targets}")
    # Validate all generated targets
    validation_results = env.validate_target_position(robust_targets)
    print(f"Validation results: {validation_results}")
    print(f"All targets valid: {validation_results.all().item()}")
    
    # Test 4: Test fallback generation
    print("\n4. Testing fallback target generation:")
    fallback_targets = env.generate_fallback_targets(start_positions, 2.0, 8.0)
    print(f"Fallback targets: {fallback_targets}")
    
    # Test 5: Test the complete reset process
    print("\n5. Testing complete reset process with validation:")
    env.reset()
    # Check if all targets are valid after reset
    final_validation = env.validate_target_position(env.target)
    print(f"Final validation after reset: {final_validation}")
    print(f"All targets valid after reset: {final_validation.all().item()}")
    
    print("\n" + "=" * 60)
    print("Robust target validation test completed!")

def test_bptt_training_conditions():
    """Test the exact conditions used in BPTT training to verify the spawn area fix"""
    
    print("\n" + "=" * 80)
    print("TESTING BPTT TRAINING CONDITIONS (150 agents)")
    print("=" * 80)
    
    def test_spawn_area(spawn_half, description, max_attempts=1000):
        """Test a specific spawn area configuration"""
        print(f"\nTesting {description}:")
        
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
        
        try:
            env = NavigationEnv3(
                num_agent_per_scene=150,  # Same as BPTT
                random_kwargs=random_kwargs,
                scene_kwargs={"path": "VisFly/datasets/visfly-beta/configs/scenes/box15_wall_pillar"},
                visual=True,  # Need visual for collision detection
                requires_grad=False,  # Disable gradients for testing
                max_episode_steps=256,
                device="cpu",
            )
            
            print("Resetting environment...")
            env.reset()
            spawn_valid = env.validate_spawn_position(env.position)
            valid_spawns = spawn_valid.sum().item()
            print(f"Valid spawn positions: {valid_spawns}/150")
            
            # Try to generate targets
            print("Attempting target generation...")
            start_time = time.time()
            
            try:
                targets = env.generate_valid_targets(
                    start_positions=env.position,
                    num_targets=env.num_envs,
                    min_distance=2.0,
                    max_distance=8.0,
                    max_attempts=max_attempts
                )
                target_valid = env.validate_target_position(targets)
                valid_targets = target_valid.sum().item()
                print(f"Valid targets generated: {valid_targets}/150")
                success = True
            except Exception as e:
                print(f"Target generation failed: {e}")
                valid_targets = 0
                success = False
            
            end_time = time.time()
            time_taken = end_time - start_time
            print(f"Time taken: {time_taken:.2f}s")
            
            # Clean up
            del env
            
            return {
                'valid_spawns': valid_spawns,
                'valid_targets': valid_targets,
                'time_taken': time_taken,
                'success': success
            }
            
        except Exception as e:
            print(f"Environment creation failed: {e}")
            return {
                'valid_spawns': 0,
                'valid_targets': 0,
                'time_taken': 0,
                'success': False
            }
    
    # Test OLD problematic spawn area (36x36m)
    print("\n1. Testing OLD problematic spawn area (36x36m):")
    old_results = test_spawn_area([18., 18., 1.], "OLD problematic spawn area", max_attempts=1000)
    
    # Add delay to avoid OpenGL conflicts
    time.sleep(2)
    
    # Test NEW fixed spawn area (16x16m)
    print("\n2. Testing NEW fixed spawn area (16x16m):")
    new_results = test_spawn_area([8., 8., 1.], "NEW fixed spawn area", max_attempts=1000)
    
    # Summary comparison
    print("\n3. COMPARISON SUMMARY:")
    print("=" * 50)
    print(f"OLD (36x36m): {old_results['valid_spawns']}/150 valid spawns, {old_results['valid_targets']}/150 valid targets")
    print(f"NEW (16x16m): {new_results['valid_spawns']}/150 valid spawns, {new_results['valid_targets']}/150 valid targets")
    
    spawn_improvement = new_results['valid_spawns'] - old_results['valid_spawns']
    target_improvement = new_results['valid_targets'] - old_results['valid_targets']
    speed_improvement = old_results['time_taken'] - new_results['time_taken']
    
    print(f"Spawn improvement: +{spawn_improvement} more valid spawn positions")
    print(f"Target improvement: +{target_improvement} more valid targets")
    print(f"Speed improvement: {speed_improvement:.2f}s faster")
    
    # Success criteria
    spawn_success = new_results['valid_spawns'] > old_results['valid_spawns']
    target_success = new_results['valid_targets'] > old_results['valid_targets']
    
    print("\n4. ANALYSIS:")
    print("=" * 50)
    if spawn_success and target_success:
        print("✅ FIX SUCCESSFUL: New spawn area produces more valid positions and targets!")
    elif spawn_success:
        print("⚡ PARTIAL SUCCESS: More valid spawns but target generation still struggles")
    elif new_results['valid_spawns'] >= 120:  # 80% success rate
        print("⚡ ACCEPTABLE: New spawn area provides adequate success rate")
    else:
        print("❌ FIX NEEDED: New spawn area doesn't improve the situation enough")
    
    # Specific analysis for BPTT training
    print("\n5. BPTT TRAINING READINESS:")
    print("=" * 50)
    if new_results['valid_spawns'] >= 120:  # 80% success rate
        print("✅ BPTT READY: Spawn success rate is acceptable for training")
    else:
        print("⚠️  BPTT WARNING: Low spawn success rate may affect training")
    
    if new_results['valid_targets'] >= 120:  # 80% success rate
        print("✅ TARGET READY: Target generation success rate is acceptable")
    else:
        print("⚠️  TARGET WARNING: Low target success rate may cause training issues")
    
    # Calculate success rates
    spawn_rate = (new_results['valid_spawns'] / 150) * 100
    target_rate = (new_results['valid_targets'] / 150) * 100
    
    print(f"\nFinal Success Rates:")
    print(f"Spawn success rate: {spawn_rate:.1f}%")
    print(f"Target success rate: {target_rate:.1f}%")
    
    if spawn_rate >= 80 and target_rate >= 80:
        print("🎉 EXCELLENT: Ready for BPTT training!")
    elif spawn_rate >= 60 and target_rate >= 60:
        print("👍 GOOD: Acceptable for BPTT training")
    else:
        print("⚠️  NEEDS IMPROVEMENT: May struggle during training")
    
    print("\n" + "=" * 80)
    print("BPTT training conditions test completed!")

if __name__ == "__main__":
    import time
    
    # Run original robust validation test
    test_robust_target_validation()
    
    # Run new BPTT conditions test
    test_bptt_training_conditions() 