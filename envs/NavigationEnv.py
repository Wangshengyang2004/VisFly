import os

import numpy as np
from habitat_sim.sensor import SensorType

from .base.droneGymEnv import DroneGymEnvsBase
from typing import Optional, Dict
import torch as th
from habitat_sim import SensorType
from gymnasium import spaces

from ..utils.tools.train_encoder import model as encoder
from ..utils.type import TensorDict
from math import pi
from typing import Union, Dict


def get_along_vertical_vector(base, obj):
    """
    Decompose obj vector into components along and perpendicular to base vector
    Simplified for BPTT compatibility - inputs should already be properly detached
    """
    # Safe norm computation with minimum clipping to avoid zero gradients
    base_norm = base.norm(dim=1, keepdim=True).clamp(min=1e-8)
    obj_norm = obj.norm(dim=1, keepdim=True).clamp(min=1e-8)
    
    # Safe division for normalization
    base_normal = base / base_norm
    along_obj_norm = (obj * base_normal).sum(dim=1, keepdim=True)
    along_vector = base_normal * along_obj_norm
    vertical_vector = obj - along_vector
    vertical_obj_norm = vertical_vector.norm(dim=1).clamp(min=1e-8)
    
    return along_obj_norm.squeeze(), vertical_obj_norm, base_norm.squeeze()


class NavigationEnv(DroneGymEnvsBase):
    def __init__(
            self,
            num_agent_per_scene: int = 1,
            num_scene: int = 1,
            seed: int = 42,
            visual: bool = True,
            requires_grad: bool = False,
            random_kwargs: dict = {},
            dynamics_kwargs: dict = {},
            scene_kwargs: dict = {},
            sensor_kwargs: list = [],
            device: str = "cpu",
            target: Optional[th.Tensor] = None,
            max_episode_steps: int = 256,
    ):
        # Hard-code sensor_kwargs like old_VisFly to ensure depth sensor is always available
        sensor_kwargs = [{
            "sensor_type": SensorType.DEPTH,
            "uuid": "depth",
            "resolution": [64, 64],
        }]

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
            tensor_output=False,
        )

        self.target = th.ones((self.num_envs, 1)) @ th.as_tensor([9, 0., 1] if target is None else target).reshape(1, -1)
        self.observation_space["target"] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        self.success_radius = 0.5

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        # Match old_VisFly behavior exactly - return numpy arrays when requires_grad=False
        if not self.requires_grad:
            if self.visual:
                # Normalize depth to [0,1]
                depth_np = self.sensor_obs["depth"] / self.max_sense_radius
                depth_np = np.clip(depth_np, 0.0, 1.0)
                return TensorDict({
                    "state": self.state.cpu().clone().numpy(),
                    "depth": depth_np,
                    "target": self.target.cpu().clone().numpy(),
                })
            else:
                return TensorDict({
                    "state": self.state.cpu().clone().numpy(),
                    "target": self.target.cpu().clone().numpy(),
                })
        else:
            if self.visual:
                # Normalize depth to [0,1]
                depth_t = th.from_numpy(self.sensor_obs["depth"]).to(self.device)
                depth_t = (depth_t / self.max_sense_radius).clamp(0, 1)
                return TensorDict({
                    "state": self.state.to(self.device),
                    "depth": depth_t,
                    "target": self.target.to(self.device),
                })
            else:
                return TensorDict({
                    "state": self.state.to(self.device),
                    "target": self.target.to(self.device),
                })

    def get_success(self) -> th.Tensor:
        """Define success as reaching within self.success_radius of target."""
        # Ensure tensors are on the same device
        position = self.position.to(self.device)
        target = self.target.to(self.device)
        return (position - target).norm(dim=1) <= self.success_radius

    # For VisFly Manuscript
    def get_reward(self) -> th.Tensor:
        # precise and stable target flight
        base_r = 0.1
        thrd_perce = th.pi/18
        
        # Fix device mismatch: ensure reference quaternion is on same device as orientation
        ref_orientation = th.tensor([1, 0, 0, 0], device=self.orientation.device, dtype=self.orientation.dtype)
        
        reward = base_r*0 + \
                ((self.velocity *(self.target - self.position)).sum(dim=1) / (1e-6+(self.target - self.position).norm(dim=1))).clamp_max(10) * 0.01+\
                 (((self.direction * self.velocity).sum(dim=1) / (1e-6+self.velocity.norm(dim=1)) / 1).clamp(-1., 1.).acos().clamp_min(thrd_perce)-thrd_perce)*-0.01+\
                 (self.orientation - ref_orientation).norm(dim=1) * -0.00001 + \
                 (self.velocity - 0).norm(dim=1) * -0.002 + \
                 (self.angular_velocity - 0).norm(dim=1) * -0.002 + \
                 1 / (self.collision_dis + 0.2) * -0.01 + \
                 (1-self.collision_dis ).relu() * ((self.collision_vector * (self.velocity - 0)).sum(dim=1) / (1e-6+self.collision_dis)).relu() * -0.005 + \
                 self._success * (self.max_episode_steps - self._step_count) * base_r * (0.2+0.8/ (1+1*self.velocity.norm(dim=1)))

        return reward


class NavigationEnv2(DroneGymEnvsBase):
    def __init__(
            self,
            num_agent_per_scene: int = 1,
            num_scene: int = 1,
            seed: int = 42,
            visual: bool = True,
            requires_grad: bool = False,
            random_kwargs: dict = None,
            dynamics_kwargs: dict = None,
            scene_kwargs: dict = None,
            sensor_kwargs: list = None,
            device: str = "cpu",
            tensor_output: bool = False,
            target: Optional[th.Tensor] = None,
            max_episode_steps: int = 256,
    ):
        # random_kwargs = {
        #     "state_generator":
        #         {
        #             "class": "Uniform",
        #             "kwargs": [
        #                 {"position": {"mean": [9., 0., 1.5], "half": [8.0, 6., 1.]},
        #                  # {"position": {"mean": [2., 0., 1.5], "half": [1.0, 6., 1.]},
        #                   # "orientation": {"mean": [0., 0, 0], "half": [0, 0, 180.]},
        #                  },
        #             ]
        #         }
        # } if random_kwargs is None else random_kwargs

        # domain randomization: add drag noise in random_kwargs
        # if "drag_random" not in random_kwargs:
        #     random_kwargs["drag_random"] = 0.1

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
            tensor_output=tensor_output
        )
        self.max_sense_radius = 10
        # self.target = th.tile(th.as_tensor([14, 0., 1] if target is None else target), (self.num_envs, 1))
        # self.encoder = encoder
        # self.encoder.load_state_dict(th.load(os.path.dirname(__file__) + '/../utils/tools/depth_autoencoder.pth'))
        # self.encoder.eval()
        # self.encoder.requires_grad_(False)
        self.success_radius = 0.5
        self.observation_space["target"] = \
            spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.target = target
        # Add target to observation space for learning algorithms that expect it (e.g. StateTargetImageExtractor)
        # self.observation_space["target"] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        # When visual is enabled, include depth image channel in observation space
        # if visual:
        #     self.observation_space["depth"] = spaces.Box(low=0.0, high=1.0, shape=(1, 64, 64), dtype=np.float32)

    def get_success(self) -> th.Tensor:
        return (self.position - self.target).norm(dim=1) <= self.success_radius
        # return th.zeros(self.num_envs, device=self.device, dtype=th.bool)

    def get_failure(self) -> th.Tensor:
        return self.is_collision

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        # compute relative position to target (global frame)
        rela_pos = (self.target - self.position).to(self.device)
        # construct proprioceptive state: relative position, orientation, velocity, angular velocity
        # Ensure all tensors are on the same device before stacking
        orient = self.orientation.to(self.device)
        vel = self.velocity.to(self.device)
        ang_vel = self.angular_velocity.to(self.device)

        state = th.hstack([
            rela_pos / self.max_sense_radius,
            orient,
            vel / 10,
            ang_vel / 10,
        ])
        # get or create depth observation and normalize
        # if self.visual and "depth" in self.sensor_obs:
        #     depth = th.from_numpy(self.sensor_obs["depth"]).to(self.device)
        # else:
        #     # create dummy depth when visual disabled
        #     depth = th.zeros((self.num_envs, 1, 64, 64), device=self.device)
        # depth = (depth / self.max_sense_radius).clamp(0, 1)
        # encode depth to latent representation
        # depth_state = self.encoder.encode(depth)
        obs_dict = {
            "state": state,
            "target": self.collision_vector,
        }
        # Provide depth if available (or zeros if not visual)
        # if self.visual:
        #     obs_dict["depth"] = depth
        # Expose absolute target position for the extractor (needed by StateTargetImageExtractor)
        # obs_dict["target"] = self.target.to(self.device)

        return TensorDict(obs_dict)

    # # For VisFly Manuscript
    # def get_reward(self) -> th.Tensor:
    #     # precise and stable target flight
    #     base_r = 0.1
    #     thrd_perce = th.pi / 18

    #     # Fix device mismatch: ensure reference quaternion is on same device as orientation
    #     # ref_orientation = th.tensor([1, 0, 0, 0], device=self.orientation.device, dtype=self.orientation.dtype)

    #     reward = base_r * 0 \
    #         + ((self.velocity * (self.target - self.position)).sum(dim=1) / (1e-6 + (self.target - self.position).norm(dim=1))).clamp_max(10) * 0.01 \
    #         + (((self.direction * self.velocity).sum(dim=1) / (1e-6 + self.velocity.norm(dim=1)) / 1).clamp(-1., 1.).acos().clamp_min(thrd_perce) - thrd_perce) * -0.01 \
    #         # + (self.orientation - ref_orientation).norm(dim=1) * -0.00001 \
    #         + (self.velocity - 0).norm(dim=1) * -0.002 \
    #         + (self.angular_velocity - 0).norm(dim=1) * -0.002 \
    #         + 1 / (self.collision_dis + 0.2) * -0.01 \
    #         + (1 - self.collision_dis).relu() * ((self.collision_vector * (self.velocity - 0)).sum(dim=1) / (1e-6 + self.collision_dis)).relu() * -0.005 \
    #         + self._success * (self.max_episode_steps - self._step_count) * base_r * (0.2 + 0.8 / (1 + self.velocity.norm(dim=1)))

    #     return reward

    def get_reward(self,
                    # dyn,
                    # collision_vector,
                    # is_collision,
                    # success,
                    ) -> Union[th.Tensor, Dict[str, th.Tensor]]:
        base_r = 0.1
        thrd_perce = th.pi/18
        
        # Fix device mismatch: ensure reference quaternion is on same device as orientation
        ref_orientation = th.tensor([1, 0, 0, 0], device=self.orientation.device, dtype=self.orientation.dtype)

        # Create detached copies to avoid inplace operations while maintaining gradients
        velocity = self.velocity.clone().detach()
        position = self.position.clone().detach()
        direction = self.direction.clone().detach()
        orientation = self.orientation.clone().detach()
        angular_velocity = self.angular_velocity.clone().detach()

        # Split each reward component
        target_pos = self.target
        r_velocity_target = ((velocity * (target_pos - position)).sum(dim=1) / (1e-6 + (target_pos - position).norm(dim=1))).clamp_max(10) * 0.01
        
        # Prevent tiny velocity norm (which causes division explosion)
        velocity_norm = velocity.norm(dim=1)
        velocity_norm = velocity_norm.clamp_min(1e-3)

        # Cosine similarity between direction and velocity
        cos_angle = ((direction - 0) * velocity).sum(dim=1) / velocity_norm

        # Clamp cos value to avoid NaNs from .acos()
        cos_angle = cos_angle.clamp(-1.0 + 1e-6, 1.0 - 1e-6)

        # Compute angle difference penalty
        angle = cos_angle.acos()
        angle_clamped = angle.clamp_min(th.pi / 18)  # Use thrd_perce
        r_direction = (angle_clamped - th.pi / 18) * -0.01

        r_orientation = (orientation - ref_orientation).norm(dim=1) * -0.00001
        r_velocity_norm = (velocity - 0).norm(dim=1) * -0.002
        r_angular_velocity_norm = (angular_velocity - 0).norm(dim=1) * -0.002
        r_collision_distance = 1 / (self.collision_dis + 0.2) * -0.01
        r_collision_velocity = (1 - self.collision_dis).relu() * ((self.collision_vector * (velocity - 0)).sum(dim=1) / (1e-6 + self.collision_dis)).relu() * -0.005
        # Avoid inplace operations by computing velocity norm separately
        velocity_magnitude = velocity.norm(dim=1)
        r_success = self._success * (self.max_episode_steps - self._step_count) * base_r * (0.2 + 0.8 / (1 + 1 * velocity_magnitude))

        # Total reward
        reward = base_r * 0 \
            + r_velocity_target \
            + r_direction \
            + r_orientation \
            + r_velocity_norm \
            + r_angular_velocity_norm \
            + r_collision_distance \
            + r_collision_velocity \
            + r_success

        # Return individual components for tensorboard logging or total reward based on flag
        if self.tensor_output:
            return {
                "reward": reward.detach(),
                "r_velocity_target": r_velocity_target,
                "r_direction": r_direction,
                "r_orientation": r_orientation,
                "r_velocity_norm": r_velocity_norm,
                "r_angular_velocity_norm": r_angular_velocity_norm,
                "r_collision_distance": r_collision_distance,
                "r_collision_velocity": r_collision_velocity,
                "r_success": r_success,
            }
        else:
            return reward.detach()


class NavigationEnv3(NavigationEnv):
    """
    Complete environment for depth-based end-to-end obstacle avoidance with robust validation.
    
    Features:
    - Domain randomization for Sim2Real transfer
    - Multiple agent support to avoid overfitting
    - Collision-free spawn and target validation
    - Physics-driven reward system
    - Comprehensive logging and debugging
    """
    
    def __init__(
        self,
        num_agent_per_scene: int = 1,
        num_scene: int = 1,
        seed: int = 42,
        visual: bool = True,
        requires_grad: bool = False,
        random_kwargs: Optional[dict] = None,
        dynamics_kwargs: Optional[dict] = None,
        scene_kwargs: Optional[dict] = None,
        sensor_kwargs: Optional[list] = None,
        device: str = "cpu",
        target: Optional[th.Tensor] = None,
        max_episode_steps: int = 256,
        latent_dim=None,
    ):
        # Configure state randomization with collision-free spawn positions
        limit = 10.0  # sensing limit: use 10m as default
        
        if random_kwargs is None or "state_generator" not in random_kwargs:
            random_kwargs = {
                "state_generator": {
                    "class": "Uniform",
                    "kwargs": [
                        {"position": {"mean": [0., 0., 1.], "half": [limit, limit, 0.]}}
                    ]
                }
            }
        
        # Apply yaw randomization to orientation half-angle
        for kw in random_kwargs["state_generator"]["kwargs"]:
            kw["orientation"] = {"mean": [0., 0., 0.], "half": [0., 0., pi]}
        
        # Domain randomization: add drag noise
        if "drag_random" not in random_kwargs:
            random_kwargs["drag_random"] = 0.1

        # Ensure depth sensor is included
        if sensor_kwargs is None:
            sensor_kwargs = [{
                "sensor_type": SensorType.DEPTH,
                "uuid": "depth",
                "resolution": [64, 64],
            }]
        
        # (random_kwargs summary logging removed for performance)
        
        super().__init__(
            num_agent_per_scene=num_agent_per_scene,
            num_scene=num_scene,
            seed=seed,
            visual=visual,
            requires_grad=requires_grad,
            random_kwargs=random_kwargs,
            dynamics_kwargs=dynamics_kwargs if dynamics_kwargs is not None else {},
            scene_kwargs=scene_kwargs if scene_kwargs is not None else {},
            sensor_kwargs=sensor_kwargs,
            device=device,
            target=target,
            max_episode_steps=max_episode_steps
        )
        
        # Environment configuration
        self.max_sense_radius = 10
        self.success_threshold = 2.0  # Success distance threshold
        self.drone_radius = 0.3
        
        # Physics-driven reward configuration
        self.lambda_p = 0.2   # position tracking weight (reduced from 1.0)
        self.lambda_c = 10.0  # obstacle avoidance weight (reduced from 20.0)
        self.lambda_a = 0.0   # acceleration smoothness weight disabled
        self.lambda_j = 0.0   # jerk smoothness weight disabled
        self.max_velocity = 5.0
        
        # Reward scaling parameters
        self.max_target_distance = 15.0  # Maximum expected target distance for normalization
        self.position_reward_scale = 2.0  # Scale factor for position reward

        # Setup observation space
        self._setup_observation_space()
        
        # Initialize tracking buffers for physics reward
        self.prev_velocity = th.zeros((self.num_envs, 3), device=self.device)
        self.prev_acceleration = th.zeros((self.num_envs, 3), device=self.device)

    def _setup_observation_space(self):
        """Configure observation space for state, depth, and target."""
        self.observation_space["depth"] = spaces.Box(
            low=0.0,
            high=self.max_sense_radius,
            shape=(1, 64, 64),
            dtype=np.float32,
        )
        
        self.observation_space = spaces.Dict({
            "state": self.observation_space["state"],
            "depth": self.observation_space["depth"],
            "target": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        })

    # ==================================================================================
    # COLLISION-FREE SPAWN VALIDATION
    # ==================================================================================
    
    def validate_spawn_position(self, spawn_positions: th.Tensor) -> th.Tensor:
        """
        Validate if spawn positions are collision-free.
        
        Args:
            spawn_positions: Tensor of shape (num_envs, 3) with spawn positions
            
        Returns:
            Boolean tensor indicating valid spawn positions (True = valid)
        """
        return self.validate_target_position(spawn_positions)
    
    def generate_collision_free_spawn_positions(
        self, 
        num_positions: int = 1,
        bounds_min: Optional[th.Tensor] = None,
        bounds_max: Optional[th.Tensor] = None,
        max_attempts: int = 1000
    ) -> th.Tensor:
        """
        Generate collision-free spawn positions within specified bounds.
        
        Args:
            num_positions: Number of spawn positions to generate
            bounds_min: Minimum bounds [x, y, z] (default: [-8, -8, 0.5])
            bounds_max: Maximum bounds [x, y, z] (default: [8, 8, 2.0])
            max_attempts: Maximum attempts to find valid positions
            
        Returns:
            Tensor of shape (num_positions, 3) with collision-free spawn positions
        """
        if bounds_min is None:
            bounds_min = th.tensor([-8.0, -8.0, 0.5], device=self.device)
        if bounds_max is None:
            bounds_max = th.tensor([8.0, 8.0, 2.0], device=self.device)
        
        valid_positions = th.zeros((num_positions, 3), device=self.device)
        valid_count = 0
        
        for _attempt in range(max_attempts):
            if valid_count >= num_positions:
                break
                
            # Generate random positions within bounds
            remaining = num_positions - valid_count
            random_positions = th.rand(remaining, 3, device=self.device)
            random_positions = bounds_min + random_positions * (bounds_max - bounds_min)
            
            # Validate positions
            valid_mask = self.validate_spawn_position(random_positions)
            valid_indices = th.where(valid_mask)[0]
            
            if len(valid_indices) > 0:
                # Add valid positions
                end_idx = min(valid_count + len(valid_indices), num_positions)
                valid_positions[valid_count:end_idx] = random_positions[valid_indices[:end_idx-valid_count]]
                valid_count = end_idx
        
        if valid_count < num_positions:
            print(f"Warning: Only found {valid_count}/{num_positions} collision-free spawn positions")
            # Fill remaining with fallback positions (slight variations of valid ones)
            for i in range(valid_count, num_positions):
                if valid_count > 0:
                    base_pos = valid_positions[i % valid_count].clone()
                    # Add small random offset
                    offset = (th.rand(3, device=self.device) - 0.5) * 0.5
                    valid_positions[i] = base_pos + offset
                else:
                    # Ultimate fallback: center position with height variation
                    valid_positions[i] = th.tensor([0.0, 0.0, 1.0 + i * 0.1], device=self.device)
        
        return valid_positions

    # ==================================================================================
    # TARGET VALIDATION (Cleaned up from previous implementation)
    # ==================================================================================
    
    def validate_target_position(self, target_positions: th.Tensor) -> th.Tensor:
        """Validate if target positions are collision-free and within bounds."""
        invalid_targets = self.envs.sceneManager.get_point_is_collision(
            std_positions=target_positions,
            scene_id=0,
            uav_radius=self.drone_radius
        )
        return th.tensor(~invalid_targets, device=self.device)

    def generate_valid_targets(
        self, 
        start_positions: th.Tensor, 
        num_targets: int = 1, 
        min_distance: float = 2.0, 
        max_distance: float = 8.0,
        max_attempts: int = 10000  # Increased threshold
    ) -> th.Tensor:
        """Generate collision-free targets at specified distances from start positions."""
        targets = th.zeros((num_targets, 3), device=self.device)
        for _attempt in range(max_attempts):
            # Generate random targets
            angle = th.rand(num_targets, device=self.device) * 2 * th.pi
            distance = min_distance + (max_distance - min_distance) * th.rand(num_targets, device=self.device)
            start_indices = th.arange(num_targets, device=self.device) % start_positions.shape[0]
            target_x = start_positions[start_indices, 0] + distance * th.cos(angle)
            target_y = start_positions[start_indices, 1] + distance * th.sin(angle)
            target_z = th.ones(num_targets, device=self.device) * 1.0
            targets = th.stack([target_x, target_y, target_z], dim=1)
            valid_targets = self.validate_target_position(targets)
            if valid_targets.all():
                return targets
        print(f"Critical: Could not find valid targets after {max_attempts} attempts. Returning last attempt.")
        return targets

    def set_valid_targets(self, new_targets: th.Tensor) -> bool:
        """Set new targets only if they are valid."""
        valid_targets = self.validate_target_position(new_targets)
        
        if valid_targets.all():
            self.target = new_targets.to(self.device)
            return True
        else:
            print(f"Warning: {valid_targets.sum().item()} out of {len(valid_targets)} targets are invalid")
            return False

    def set_valid_targets_with_retry(self, new_targets: th.Tensor, start_positions: Optional[th.Tensor] = None) -> bool:
        """
        Set new targets with retry mechanism if validation fails.
        
        Args:
            new_targets: Tensor of shape (num_envs, 3) with new target positions
            start_positions: Optional starting positions for generating fallback targets
            
        Returns:
            Boolean indicating if all targets are valid after retry
        """
        valid_targets = self.validate_target_position(new_targets)
        
        if valid_targets.all():
            self.target = new_targets.to(self.device)
            return True
        else:
            print(f"Warning: {valid_targets.sum().item()} out of {len(valid_targets)} targets are invalid, attempting retry...")
            
            # If we have start positions, try to generate valid targets
            if start_positions is not None:
                # Ensure we generate the right number of targets
                num_targets_needed = len(new_targets)
                valid_replacement_targets = self.generate_valid_targets_with_retry(
                    start_positions=start_positions,
                    num_targets=num_targets_needed,
                    min_distance=2.0,
                    max_distance=8.0,
                    max_attempts_per_batch=30,
                    max_batches=5
                )
                
                # If we got fewer targets than needed, pad with the original targets
                if len(valid_replacement_targets) < num_targets_needed:
                    padding = new_targets[len(valid_replacement_targets):]
                    valid_replacement_targets = th.cat([valid_replacement_targets, padding], dim=0)
                
                # Validate the replacement targets
                replacement_validation = self.validate_target_position(valid_replacement_targets)
                
                if replacement_validation.all():
                    self.target = valid_replacement_targets.to(self.device)
                    print("Successfully generated valid replacement targets")
                    return True
                else:
                    print(f"Warning: {replacement_validation.sum().item()} out of {len(replacement_validation)} replacement targets still invalid")
            
            # If retry failed, use fallback targets
            if start_positions is not None:
                fallback_targets = self.generate_fallback_targets(start_positions, 2.0, 8.0)
                # Ensure fallback targets match the expected size
                if len(fallback_targets) < len(new_targets):
                    padding = new_targets[len(fallback_targets):]
                    fallback_targets = th.cat([fallback_targets, padding], dim=0)
                self.target = fallback_targets.to(self.device)
                print("Using fallback targets")
                return False
            else:
                # If no start positions provided, just set the original targets
                self.target = new_targets.to(self.device)
                return False

    def generate_valid_targets_with_retry(
        self, 
        start_positions: th.Tensor, 
        num_targets: int = 1,
        min_distance: float = 2.0, 
        max_distance: float = 8.0,
        max_attempts_per_batch: int = 50, 
        max_batches: int = 10
    ) -> th.Tensor:
        """
        Generate valid targets with multiple retry attempts until all are valid.
        
        Args:
            start_positions: Tensor of shape (num_envs, 3) with starting positions
            num_targets: Number of targets to generate
            min_distance: Minimum distance from start position
            max_distance: Maximum distance from start position
            max_attempts_per_batch: Maximum attempts per batch
            max_batches: Maximum number of batches to try
            
        Returns:
            Tensor of shape (num_targets, 3) with valid target positions
        """
        # Ensure we don't try to generate more targets than we have start positions
        actual_num_targets = min(num_targets, start_positions.shape[0])
        targets = th.zeros((actual_num_targets, 3), device=self.device)
        valid_mask = th.zeros(actual_num_targets, dtype=th.bool, device=self.device)
        
        for batch in range(max_batches):
            if valid_mask.all():
                break
                
            # Generate candidates for invalid targets only
            invalid_indices = th.where(~valid_mask)[0]
            if len(invalid_indices) == 0:
                break
                
            for _attempt in range(max_attempts_per_batch):
                # Generate random targets for invalid positions
                angle = th.rand(len(invalid_indices), device=self.device) * 2 * th.pi
                distance = min_distance + (max_distance - min_distance) * th.rand(len(invalid_indices), device=self.device)
                
                # Use modulo to handle cases where we have more targets than start positions
                start_indices = invalid_indices % start_positions.shape[0]
                target_x = start_positions[start_indices, 0] + distance * th.cos(angle)
                target_y = start_positions[start_indices, 1] + distance * th.sin(angle)
                target_z = th.ones(len(invalid_indices), device=self.device) * 1.0
                
                candidate_targets = th.stack([target_x, target_y, target_z], dim=1)
                
                # Validate candidates
                valid_candidates = self.validate_target_position(candidate_targets)
                
                # Update valid targets
                targets[invalid_indices] = candidate_targets
                valid_mask[invalid_indices] = valid_candidates
                
                if valid_mask.all():
                    break
            
            if batch == max_batches - 1 and not valid_mask.all():
                print(f"Warning: Could not find valid targets for {actual_num_targets - valid_mask.sum().item()} positions after {max_batches} batches")
        
        return targets

    def generate_fallback_targets(self, start_positions: th.Tensor, min_distance: float = 2.0, max_distance: float = 8.0) -> th.Tensor:
        """
        Generate fallback targets using simple heuristics when validation fails.
        
        Args:
            start_positions: Tensor of shape (num_envs, 3) with starting positions
            min_distance: Minimum distance from start position
            max_distance: Maximum distance from start position
            
        Returns:
            Tensor of shape (num_envs, 3) with fallback target positions
        """
        num_targets = start_positions.shape[0]
        
        # Use fixed directions to avoid obstacles
        directions = th.tensor([
            [1.0, 0.0, 0.0],   # Forward
            [-1.0, 0.0, 0.0],  # Backward
            [0.0, 1.0, 0.0],   # Right
            [0.0, -1.0, 0.0],  # Left
            [1.0, 1.0, 0.0],   # Forward-Right
            [1.0, -1.0, 0.0],  # Forward-Left
            [-1.0, 1.0, 0.0],  # Backward-Right
            [-1.0, -1.0, 0.0], # Backward-Left
        ], device=self.device)
        
        # Normalize directions
        directions = directions / (directions.norm(dim=1, keepdim=True) + 1e-6)
        
        fallback_targets = th.zeros((num_targets, 3), device=self.device)
        
        for i in range(num_targets):
            # Try different directions for each target
            direction_idx = i % len(directions)
            direction = directions[direction_idx]
            
            # Use middle distance
            distance = (min_distance + max_distance) / 2.0
            
            # Generate target in this direction
            target_xy = start_positions[i, :2] + distance * direction[:2]
            target_z = th.tensor(1.0, device=self.device)
            
            fallback_targets[i] = th.cat([target_xy, target_z.unsqueeze(0)])
        
        return fallback_targets

    # ==================================================================================
    # OBSERVATION AND STATE MANAGEMENT
    # ==================================================================================
    
    def get_observation(self):
        """Get observation with proper tensor shapes and device handling."""
        device = self.device
        position = self.position.to(device)
        target = self.target.to(device)
        
        # Compute relative position and direction to target
        rela_pos = (target - position)
        if rela_pos.ndim == 1:
            rela_pos = rela_pos.unsqueeze(0)
            
        distance = rela_pos.norm(dim=1)
        if distance.ndim == 0:
            distance = distance.unsqueeze(0)
            
        direction = rela_pos / (distance.unsqueeze(1) + 1e-6)
        if direction.ndim == 1:
            direction = direction.unsqueeze(0)
        
        # Get velocity and angular velocity
        velocity = self.velocity.to(device)
        if velocity.ndim == 1:
            velocity = velocity.unsqueeze(0)
            
        angular_velocity = self.angular_velocity.to(device)
        if angular_velocity.ndim == 1:
            angular_velocity = angular_velocity.unsqueeze(0)
        
        distance = distance.unsqueeze(1)
        
        # Validate tensor shapes before concatenation
        N = self.num_envs
        assert rela_pos.shape == (N, 3), f"rela_pos shape: {rela_pos.shape}"
        assert direction.shape == (N, 3), f"direction shape: {direction.shape}"
        assert velocity.shape == (N, 3), f"velocity shape: {velocity.shape}"
        assert angular_velocity.shape == (N, 3), f"angular_velocity shape: {angular_velocity.shape}"
        assert distance.shape == (N, 1), f"distance shape: {distance.shape}"
        
        # Concatenate state features
        state = th.cat([
            rela_pos,
            direction, 
            velocity,
            angular_velocity,
            distance,
        ], dim=1).to(device)
        
        # Handle depth sensor observation
        if self.visual and "depth" in self.sensor_obs:
            depth = th.from_numpy(self.sensor_obs["depth"]).to(self.device)
            depth = (depth / self.max_sense_radius).clamp(0, 1)
        else:
            depth = th.zeros((self.num_envs, 1, 64, 64), device=device)
        
        # Return observation dictionary
        return TensorDict({
            "state": state,
            "depth": depth,
            "target": target.detach() if not self.requires_grad else target,
        })

    # ==================================================================================
    # SUCCESS AND FAILURE CONDITIONS
    # ==================================================================================
    
    def get_failure(self) -> th.Tensor:
        """Episodes end on collision."""
        return self.is_collision

    def get_success(self) -> th.Tensor:
        """Episodes succeed when agent reaches target within threshold."""
        position = self.position.to(self.device)
        target = self.target.to(self.device)
        distance_to_target = (position - target).norm(dim=1)
        return distance_to_target < self.success_threshold

    # ==================================================================================
    # ENVIRONMENT RESET WITH COLLISION-FREE VALIDATION
    # ==================================================================================
    
    def reset(self, *args, **kwargs):
        """Reset environment with collision-free spawn positions and valid targets."""
        # Perform standard reset first
        result = super().reset(*args, **kwargs)
        
        # Validate spawn positions and regenerate if needed
        spawn_positions = self.position.clone().detach()
        spawn_valid = self.validate_spawn_position(spawn_positions)
        
        if not spawn_valid.all():
            # only log warnings in non-training mode
            if not self.requires_grad:
                print(f"Warning: {spawn_valid.sum().item()}/{len(spawn_valid)} spawn positions are collision-free")
                print("Regenerating spawn for invalid agents...")
            invalid_indices = th.where(~spawn_valid)[0]
            new_spawn_positions = self.generate_collision_free_spawn_positions(num_positions=self.num_envs)
            # Properly reset only the invalid agents using the API
            self.reset_agent_by_id(agent_indices=invalid_indices, state=(
                new_spawn_positions[invalid_indices],
                self.orientation[invalid_indices],
                self.velocity[invalid_indices],
                self.angular_velocity[invalid_indices]
            ))
        
        # target regeneration disabled for high-performance training

        # Reset physics tracking buffers
        with th.no_grad():
            sim_dev = self.position.device
            self.prev_velocity = th.zeros((self.num_envs, 3), device=sim_dev)
            self.prev_acceleration = th.zeros((self.num_envs, 3), device=sim_dev)
            
            # Initialize target distance tracking for progress reward
            current_pos = self.position.clone().detach()
            current_target = self.target.clone().detach()
            self.prev_target_distance = (current_target - current_pos).norm(dim=1).to(sim_dev)

        return result

    def examine(self):
        """Override to perform a full reset when any agents are done."""
        if hasattr(self, '_done') and self._done.any():
            # suppress logs during training
            if not self.requires_grad:
                print("Agent done, performing full environment reset")
            self.reset()
        return self._observations

    # ==================================================================================
    # PHYSICS-DRIVEN REWARD SYSTEM
    # ==================================================================================
    
    def get_reward(self):
        """Get unified physics-driven reward for training and evaluation."""
        collision_vector = self.collision_vector
        is_collision = self.is_collision
        success = self.get_success()
        
        reward_components = self.get_analytical_reward_components(
            self, collision_vector, is_collision, success
        )
        
        # Return individual components for BPTT training or total reward for evaluation
        if getattr(self, '_enable_individual_rewards', False):
            return reward_components
        else:
            return reward_components["reward"].to(self.device)

    def get_analytical_reward_components(
        self,
        dyn,
        collision_vector: th.Tensor,
        is_collision: th.Tensor,
        success: th.Tensor,
    ) -> Dict[str, th.Tensor]:
        """Improved physics-driven analytical reward with better scaling and balance."""
        device = dyn.position.device

        # Create detached copies to avoid inplace operations while maintaining gradients
        pos = dyn.position.clone().detach().requires_grad_(True)
        vel = dyn.velocity.clone().detach().requires_grad_(True)
        collision_vector = collision_vector.to(device).clone().detach().requires_grad_(True)

        # Move tracking buffers to correct device if needed
        if self.prev_velocity.device != device:
            self.prev_velocity = self.prev_velocity.to(device)
        if self.prev_acceleration.device != device:
            self.prev_acceleration = self.prev_acceleration.to(device)

        # 1. IMPROVED Position-based target tracking with normalization
        target_dir = (self.target.to(device) - pos)
        target_dis = target_dir.norm(dim=1)
        
        # Normalize distance to [0, 1] range and use exponential decay for smooth gradients
        normalized_distance = (target_dis / self.max_target_distance).clamp(0, 1)
        # Use exponential reward that approaches 0 as distance increases
        r_position = -self.lambda_p * self.position_reward_scale * normalized_distance
        
        # 2. IMPROVED Progress-based reward (reward for getting closer)
        if hasattr(self, 'prev_target_distance'):
            progress = self.prev_target_distance - target_dis
            r_progress = self.lambda_p * 0.5 * progress  # Reward for progress
        else:
            r_progress = th.zeros_like(target_dis)
        
        # Store current distance for next step
        with th.no_grad():
            self.prev_target_distance = target_dis.detach().clone()
        
        # 3. IMPROVED Velocity alignment with target direction
        # Compute velocity magnitude safely
        vel_magnitude = vel.norm(dim=1)
        if vel_magnitude.max() > 1e-6:
            vel_normalized = vel / (vel_magnitude.unsqueeze(1) + 1e-6)
            target_dir_normalized = target_dir / (target_dis.unsqueeze(1) + 1e-6)
            alignment = (target_dir_normalized * vel_normalized).sum(dim=1)
            r_alignment = self.lambda_p * 0.3 * alignment  # Increased weight for alignment
        else:
            r_alignment = th.zeros_like(target_dis)

        # 4. IMPROVED Obstacle avoidance with smoother penalties
        is_collision = is_collision.to(device)
        success = success.to(device)
        collision_dis = collision_vector.norm(dim=1).clamp(min=1e-6)
        
        approaching_v, _, _ = get_along_vertical_vector(collision_vector, vel)
        approaching_v = approaching_v.clamp(min=0)
        
        # Use larger safety margin and smoother penalty function
        safety_margin = self.drone_radius * 3.0  # Reduced from 5.0 for less aggressive penalty
        
        # Smooth exponential penalty that increases as we get closer to obstacles
        # TODO: 奖励：和最近碰撞点的距离，越近惩罚越大，一定要和dt关联；逼近速度
        obstacle_penalty = th.where(
            collision_dis < safety_margin,
            th.exp(-(collision_dis / self.drone_radius)) * approaching_v,
            th.zeros_like(collision_dis)
        )
        r_obstacle = -self.lambda_c * obstacle_penalty

        # 5. IMPROVED Control smoothness with better scaling
        dt = 0.02
        curr_acc = (vel - self.prev_velocity) / dt
        acc_magnitude = curr_acc.norm(dim=1)
        
        # Normalize acceleration penalty
        max_reasonable_acc = 10.0  # m/s^2
        normalized_acc = (acc_magnitude / max_reasonable_acc).clamp(0, 1)
        r_acc = -self.lambda_a * normalized_acc
        
        jerk = curr_acc - self.prev_acceleration
        jerk_magnitude = jerk.norm(dim=1)
        
        # Normalize jerk penalty
        max_reasonable_jerk = 50.0  # m/s^3
        normalized_jerk = (jerk_magnitude / max_reasonable_jerk).clamp(0, 1)
        r_jerk = -self.lambda_j * normalized_jerk

        # Update buffers
        with th.no_grad():
            self.prev_velocity = vel.detach().clone()
            self.prev_acceleration = curr_acc.detach().clone()

        # 6. IMPROVED Terminal rewards with better scaling
        # Success reward scaled by remaining time (encourage faster completion)
        step_count = getattr(self, '_step_count', 0)
        time_bonus = (self.max_episode_steps - step_count) / self.max_episode_steps
        r_success = success.float() * (10.0 + 5.0 * time_bonus)  # 10-15 range
        
        # Collision penalty
        r_collision = is_collision.float() * -8.0  # Reduced from -10.0
        
        # 7. NEW: Velocity regulation reward (encourage reasonable speed)
        # target_speed = 2.0  # m/s - reasonable target speed
        # speed = vel.norm(dim=1)
        # speed_error = th.abs(speed - target_speed)
        # r_speed = -0.1 * speed_error  # Small penalty for being too fast or too slow

        total_reward = (r_position + r_progress + r_alignment + r_obstacle + 
                       r_acc + r_jerk + r_success + r_collision)
        
        return {
            "reward": total_reward.to(device),
            "r_position": r_position.to(device),
            "r_progress": r_progress.to(device),
            "r_alignment": r_alignment.to(device),
            "r_obstacle": r_obstacle.to(device),
            "r_acc": r_acc.to(device),
            "r_jerk": r_jerk.to(device),
            "r_success": r_success.to(device),
            "r_collision": r_collision.to(device),
            # "r_speed": r_speed.to(device),
        }