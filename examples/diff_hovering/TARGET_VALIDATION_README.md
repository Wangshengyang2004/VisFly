# Target Position Validation in HoverEnv2

This document describes the new target position validation functionality added to the `HoverEnv2` environment.

## Overview

The target validation system ensures that target positions are reasonable and achievable by checking:
1. **Collision Detection**: Targets are not inside obstacles
2. **Bounds Checking**: Targets are within scene boundaries
3. **Safety Margins**: Targets maintain appropriate distance from obstacles

## New Methods

### 1. `validate_target_position(target_positions)`

Validates if target positions are reasonable (not in obstacles or out of bounds).

**Parameters:**
- `target_positions`: Tensor of shape (num_envs, 3) with target positions

**Returns:**
- Boolean tensor indicating valid targets (True = valid, False = invalid)

**Example:**
```python
# Check if targets are valid
targets = th.tensor([[1.0, 0.0, 1.5], [5.0, 2.0, 1.5]])
is_valid = env.validate_target_position(targets)
print(f"Targets are valid: {is_valid}")
```

### 2. `generate_valid_targets(start_positions, num_targets, min_distance, max_distance, max_attempts)`

Generates targets that are valid (not in obstacles or out of bounds).

**Parameters:**
- `start_positions`: Tensor of shape (num_envs, 3) with starting positions
- `num_targets`: Number of targets to generate (default: 1)
- `min_distance`: Minimum distance from start position (default: 2.0)
- `max_distance`: Maximum distance from start position (default: 8.0)
- `max_attempts`: Maximum attempts to find valid targets (default: 100)

**Returns:**
- Tensor of shape (num_envs, 3) with valid target positions

**Example:**
```python
# Generate valid targets
start_pos = env.position
valid_targets = env.generate_valid_targets(
    start_positions=start_pos,
    num_targets=env.num_envs,
    min_distance=2.0,
    max_distance=8.0,
    max_attempts=100
)
```

### 3. `set_valid_targets(new_targets)`

Sets new targets only if they are valid.

**Parameters:**
- `new_targets`: Tensor of shape (num_envs, 3) with new target positions

**Returns:**
- Boolean indicating if all targets are valid

**Example:**
```python
# Set targets only if valid
new_targets = th.tensor([[2.0, 1.0, 1.5], [3.0, -1.0, 1.5]])
success = env.set_valid_targets(new_targets)
if success:
    print("Targets set successfully")
else:
    print("Some targets are invalid")
```

## Usage Examples

### Basic Validation
```python
# Create environment
env = HoverEnv2(visual=True)
env.reset()

# Validate current target
is_valid = env.validate_target_position(env.target)
print(f"Current target is valid: {is_valid}")
```

### Generate and Set Valid Targets
```python
# Generate valid targets from current position
start_positions = env.position
valid_targets = env.generate_valid_targets(
    start_positions=start_positions,
    min_distance=2.0,
    max_distance=8.0
)

# Set the valid targets
success = env.set_valid_targets(valid_targets)
if success:
    print("Successfully set valid targets")
```

### Integration with Training
```python
# In your training loop, you can use target validation
if enable_target_randomization:
    # Generate valid targets instead of random ones
    start_positions = env.position
    valid_targets = env.generate_valid_targets(
        start_positions=start_positions,
        num_targets=env.num_envs
    )
    env.set_valid_targets(valid_targets)
```

## Testing

Run the test script to verify the functionality:

```bash
cd examples/diff_hovering
python test_target_validation.py
```

## Technical Details

The validation system uses the existing `SceneManager.get_point_is_collision()` function which:
- Uses Habitat's built-in collision detection
- Checks scene bounds automatically
- Considers the drone's radius for collision checking
- Returns both collision and bounds information

The system is designed to be efficient and integrate seamlessly with existing VisFly environments. 