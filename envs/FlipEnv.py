import torch as th
import torch.nn.functional as F
from typing import Dict
import sys
import os
sys.path.append(os.getcwd())
from .base.droneGymEnv import DroneGymEnvsBase
from ..utils.type import TensorDict
from ..utils.maths import Quaternion
import numpy as np
from gymnasium import spaces

class FlipEnv(DroneGymEnvsBase):
    """Environment for flip maneuvers with multi-level reward structure."""
    
    def __init__(
        self,
        num_agent_per_scene: int = 1,
        num_scene: int = 1,
        seed: int = 42,
        visual: bool = True,
        requires_grad: bool = False,
        random_kwargs: dict = None,
        dynamics_kwargs: dict = None,
        scene_kwargs: dict = {},
        sensor_kwargs: list = [],
        device: str = "cpu",
        max_episode_steps: int = 256,
        tensor_output: bool = False,
    ):
        random_kwargs = {
            "state_generator": {
                "class": "Uniform",
                "kwargs": [
                    {"position": {"mean": [0.0, 0.0, 1.5], "half": [0.5, 0.5, 0.2]}},
                ],
            }
        } if random_kwargs is None else random_kwargs
        
        super().__init__(
            num_agent_per_scene=num_agent_per_scene,
            num_scene=num_scene,
            seed=seed,
            visual=visual,
            requires_grad=requires_grad,
            random_kwargs=random_kwargs,
            dynamics_kwargs=dynamics_kwargs,
            scene_kwargs=scene_kwargs,
            sensor_kwargs=sensor_kwargs,
            device=device,
            max_episode_steps=max_episode_steps,
            tensor_output=tensor_output,
        )
        
        # Flip parameters
        self.command_flips = th.ones((self.num_envs,), device=self.device) * 2 * th.pi * 5 # 5 full flips
        self.initial_orientation = th.zeros((self.num_envs, 4), device=self.device)
        self.accumulated_rotation = th.zeros((self.num_envs,), device=self.device)
        self.previous_x_axis = th.zeros((self.num_envs, 3), device=self.device)
        
        # Target position for maintaining position during flip
        self.target_position = th.zeros((self.num_envs, 3), device=self.device)
        self.target_position[:, 2] = 1.5  # Default height
        
        # Track if flip has been initiated
        self.flip_initiated = th.zeros((self.num_envs,), dtype=th.bool, device=self.device)
        self.steps_since_reset = th.zeros((self.num_envs,), dtype=th.int32, device=self.device)
        
        # Update observation space to account for the command and progress dimensions
        # The base state has 13 dimensions, we add 2 for command and progress
        original_state_size = self.observation_space["state"].shape[0]
        new_state_size = original_state_size + 2
        self.observation_space["state"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(new_state_size,), dtype=np.float32
        )
        
    def reset(self, state=None, obs=None, is_test=False):
        obs = super().reset(state, obs, is_test)
        
        # Reset flip tracking
        self.accumulated_rotation[:] = 0.0
        self.initial_orientation = self.orientation.clone()
        self.flip_initiated[:] = False
        self.steps_since_reset[:] = 0
        
        # Initialize previous x-axis for rotation tracking
        qw, qx, qy, qz = self.orientation[:, 0], self.orientation[:, 1], self.orientation[:, 2], self.orientation[:, 3]
        self.previous_x_axis[:, 0] = 1 - 2 * (qy**2 + qz**2)
        self.previous_x_axis[:, 1] = 2 * (qx*qy + qw*qz)
        self.previous_x_axis[:, 2] = 2 * (qx*qz - qw*qy)
        
        # Set target position to current position
        self.target_position = self.position.clone()
        
        # Randomize commanded flips (1 to 2 full rotations) - avoiding too small values
        if not is_test:
            self.command_flips = (th.rand((self.num_envs,), device=self.device) * 1.0 + 1.0) * 2 * th.pi
        
        return obs
    
    def step(self, action, is_test=False):
        obs, reward, done, info = super().step(action, is_test)
        self.steps_since_reset += 1
        return obs, reward, done, info
    
    def get_observation(self, indices=None) -> Dict:
        # Include command and progress in observation
        command_normalized = self.command_flips / (2 * th.pi)  # Normalize to number of flips
        progress_normalized = self.accumulated_rotation / (2 * th.pi)  # Current progress in flips
        
        extended_state = th.cat([
            self.state, 
            command_normalized.unsqueeze(-1),
            progress_normalized.unsqueeze(-1)
        ], dim=-1)
        
        return TensorDict({"state": extended_state})
    
    def get_success(self) -> th.Tensor:
        # Require minimum steps to prevent instant success
        min_steps_required = 20
        sufficient_steps = self.steps_since_reset >= min_steps_required
        
        # Success when flip is completed and drone is stable
        flip_complete = self.accumulated_rotation >= (self.command_flips * 0.95)  # 95% of commanded rotation
        
        # Check if drone is upright and stable
        upright = self.orientation[:, 0] > 0.95  # w component close to 1
        low_angular_velocity = self.angular_velocity.norm(dim=1) < 0.5
        near_target = (self.position - self.target_position).norm(dim=1) < 0.5
        
        # Must have initiated flip (had significant angular velocity at some point)
        return sufficient_steps & flip_complete & upright & low_angular_velocity & near_target & self.flip_initiated
    
    def get_failure(self) -> th.Tensor:
        # Fail if collision or too far from target
        position_error = (self.position - self.target_position).norm(dim=1)
        too_far = position_error > 10.0
        too_low = self.position[:, 2] < 0.1
        
        return self.is_collision | too_far | too_low
    
    def _update_rotation_tracking(self):
        """Update accumulated rotation using proper angle tracking."""
        # Current x-axis in world frame
        qw, qx, qy, qz = self.orientation[:, 0], self.orientation[:, 1], self.orientation[:, 2], self.orientation[:, 3]
        current_x_axis = th.zeros((self.num_envs, 3), device=self.device)
        current_x_axis[:, 0] = 1 - 2 * (qy**2 + qz**2)
        current_x_axis[:, 1] = 2 * (qx*qy + qw*qz)
        current_x_axis[:, 2] = 2 * (qx*qz - qw*qy)
        
        # Calculate rotation angle around x-axis using angular velocity
        # This is more reliable than trying to extract from quaternions
        x_angular_velocity = self.angular_velocity[:, 0]
        
        # Update accumulated rotation (only positive progress counts)
        rotation_increment = x_angular_velocity * self.envs.dynamics.ctrl_dt
        self.accumulated_rotation += th.clamp(rotation_increment, min=0.0)  # Only count forward rotation
        
        # Track if flip has been initiated (significant angular velocity)
        high_angular_velocity = x_angular_velocity.abs() > 1.0  # rad/s threshold
        self.flip_initiated = self.flip_initiated | high_angular_velocity
        
        # Update previous x-axis
        self.previous_x_axis = current_x_axis.clone()
    
    def get_reward(self) -> th.Tensor:
        """
        Multi-level reward function based on compute_flip_reward from previous discussion.
        """
        # Clone tensors that are part of computational graph to avoid in-place operation issues
        angular_velocity = self.angular_velocity.clone()
        velocity = self.velocity.clone()
        
        # Update rotation tracking first
        self._update_rotation_tracking()
        
        # === POSITION REWARD ===
        relative_pos_world = self.target_position - self.position
        pos_dist = relative_pos_world.norm(dim=1)
        
        # Multi-level position rewards
        pos_reward_l0 = 1.0 / (1.0 + 1 * pos_dist)
        pos_reward_l1 = 1.0 / (1.0 + 10 * pos_dist)  # Tighter tolerance
        pos_reward = pos_reward_l0 + pos_reward_l1
        
        # === FLIP PROGRESS REWARD ===
        # Progress toward commanded rotation
        progress_ratio = th.clamp(self.accumulated_rotation / self.command_flips, 0.0, 1.0)
        progress_reward_l0 = progress_ratio
        progress_reward_l1 = progress_ratio ** 2  # Quadratic for near-completion bonus
        progress_reward = progress_reward_l0 + progress_reward_l1
        
        # === ANGULAR VELOCITY SHAPING ===
        flip_incomplete = progress_ratio < 0.9
        
        # During flip: reward x-axis rotation, penalize others
        x_ang_vel = angular_velocity[:, 0]
        yz_ang_vel = angular_velocity[:, 1:].norm(dim=1)
        
        # Desired angular velocity (positive for forward flip)
        desired_x_vel = th.where(flip_incomplete, 
                                th.ones_like(x_ang_vel) * 3.0,  # 3 rad/s during flip
                                th.zeros_like(x_ang_vel))       # 0 rad/s after flip
        
        x_vel_error = (x_ang_vel - desired_x_vel).abs()
        ang_vel_reward = 1.0 / (1.0 + x_vel_error + 2.0 * yz_ang_vel)  # Heavily penalize off-axis rotation
        
        # Penalize negative x rotation during flip
        wrong_direction_penalty = th.where(
            flip_incomplete & (x_ang_vel < 0),
            -th.ones_like(x_ang_vel) * 2.0,
            th.zeros_like(x_ang_vel)
        )
        
        # === STABILIZATION REWARD ===
        # After flip completion
        qw = self.orientation[:, 0]
        upright_reward = th.where(
            progress_ratio > 0.9,
            qw * 2.0,  # Reward upright orientation after flip
            th.zeros_like(qw)
        )
        
        # === TOTAL REWARD ===
        total_reward = (
            pos_reward * 2.0 +          # Weight position maintenance
            progress_reward * 3.0 +     # Weight flip progress
            ang_vel_reward * 1.5 +      # Weight angular velocity control
            upright_reward * 1.0 +      # Weight final stabilization
            wrong_direction_penalty     # Penalty for wrong rotation
        ) / 10.0  # Normalize
        
        # === ADDITIONAL PENALTIES ===
        # Height penalty
        height_penalty = th.where(
            self.position[:, 2] < 0.5,
            -th.ones_like(total_reward) * 1.0,
            th.zeros_like(total_reward)
        )
        
        # Time penalty to encourage faster completion
        time_penalty = -0.01  # Small constant penalty per step
        
        return total_reward + height_penalty + time_penalty
    
    def detach(self):
        """Detach FlipEnv-specific tensors from computational graph for BPTT."""
        super().detach()
        
        # Detach all flip-specific state tensors
        tensors_to_detach = [
            'accumulated_rotation', 'initial_orientation', 'target_position',
            'command_flips', 'previous_x_axis', 'flip_initiated', 'steps_since_reset'
        ]
        
        for tensor_name in tensors_to_detach:
            if hasattr(self, tensor_name):
                tensor = getattr(self, tensor_name)
                if isinstance(tensor, th.Tensor):
                    setattr(self, tensor_name, tensor.clone().detach())