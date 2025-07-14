import os
import sys

import numpy as np
from .base.droneGymEnv import DroneGymEnvsBase
from typing import Union, Tuple, List, Optional, Dict
import torch as th
from habitat_sim import SensorType
from gymnasium import spaces
from ..utils.tools.train_encoder import model as encoder
from ..utils.type import TensorDict


class HoverEnv(DroneGymEnvsBase):
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
            device: Optional[th.device] = th.device("cpu"),
            target: Optional[th.Tensor] = None,
            max_episode_steps: int = 256,
            tensor_output: bool = False,
    ):

        random_kwargs = {
            "state_generator":
                {
                    "class": "Uniform",
                    "kwargs": [
                        # {"position": {"mean": [1., 0., 1.5], "half": [0.0, 0.0, 0.0]}},
                        {"position": {"mean": [1., 0., 1.5], "half": [1.0, 1.0, 0.5]}},
                    ]
                }
        }

        super().__init__(
            num_agent_per_scene=num_agent_per_scene,
            num_scene=num_scene,
            seed=seed,
            visual=visual,
            requires_grad=requires_grad,
            random_kwargs=random_kwargs,
            dynamics_kwargs=dynamics_kwargs,
            sensor_kwargs=sensor_kwargs,
            scene_kwargs=scene_kwargs,
            device=device,
            max_episode_steps=max_episode_steps,
            tensor_output=tensor_output,

        )

        self.target = th.ones((self.num_envs, 1)) @ th.as_tensor([1, 0., 1.5] if target is None else target).reshape(1,-1)
        self.success_radius = 0.5

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        obs = TensorDict({
            "state": self.state,
        })

        # if self.latent is not None:
        #     if not self.requires_grad:
        #         obs["latent"] = self.latent.cpu().numpy()
        #     else:
        #         obs["latent"] = self.latent

        return obs

    def get_success(self) -> th.Tensor:
        return th.full((self.num_agent,), False)
        # return (self.position - self.target).norm(dim=1) < self.success_radius

    def get_reward(self) -> th.Tensor:
        base_r = 0.1
        pos_factor = -0.1 * 1/9
        reward = (
                base_r +
                 (self.position - self.target).norm(dim=1) * pos_factor +
                 (self.orientation - th.tensor([1, 0, 0, 0])).norm(dim=1) * -0.00001 +
                 (self.velocity - 0).norm(dim=1) * -0.002 +
                 (self.angular_velocity - 0).norm(dim=1) * -0.002
        )

        return reward


class HoverEnv2(HoverEnv):

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
            device: Optional[th.device] = th.device("cpu"),
            target: Optional[th.Tensor] = None,
            max_episode_steps: int = 256,
            tensor_output: bool = False,
    ):
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
            target=target,
            tensor_output=tensor_output,
        )

    def validate_target_position(self, target_positions: th.Tensor) -> th.Tensor:
        """
        Validate if target positions are reasonable (not in obstacles or out of bounds)
        
        Args:
            target_positions: Tensor of shape (num_envs, 3) with target positions
            
        Returns:
            Boolean tensor indicating valid targets (True = valid, False = invalid)
        """
        # Use the existing collision detection function from SceneManager
        invalid_targets = self.envs.sceneManager.get_point_is_collision(
            std_positions=target_positions,
            scene_id=0,  # Assuming single scene
            uav_radius=getattr(self, 'uav_radius', 0.3)  # Use drone radius for collision checking
        )
        
        # Return valid targets (invert the result) and convert to tensor
        return th.tensor(~invalid_targets, device=self.device)

    def generate_valid_targets(self, start_positions: th.Tensor, num_targets: int = 1, 
                             min_distance: float = 2.0, max_distance: float = 8.0,
                             max_attempts: int = 100) -> th.Tensor:
        """
        Generate targets that are valid (not in obstacles or out of bounds)
        
        Args:
            start_positions: Tensor of shape (num_envs, 3) with starting positions
            num_targets: Number of targets to generate
            min_distance: Minimum distance from start position
            max_distance: Maximum distance from start position
            max_attempts: Maximum attempts to find valid targets
            
        Returns:
            Tensor of shape (num_envs, 3) with valid target positions
        """
        targets = th.zeros((num_targets, 3), device=self.device)
        
        for attempt in range(max_attempts):
            # Generate random targets
            angle = th.rand(num_targets, device=self.device) * 2 * th.pi
            distance = min_distance + (max_distance - min_distance) * th.rand(num_targets, device=self.device)
            
            target_x = start_positions[:, 0] + distance * th.cos(angle)
            target_y = start_positions[:, 1] + distance * th.sin(angle)
            target_z = th.ones(num_targets, device=self.device) * 1.0
            
            candidate_targets = th.stack([target_x, target_y, target_z], dim=1)
            
            # Validate targets
            valid_targets = self.validate_target_position(candidate_targets)
            
            if valid_targets.all():
                return candidate_targets
        
        # If max attempts reached, return the last generated targets
        print(f"Warning: Could not find valid targets after {max_attempts} attempts")
        return candidate_targets

    def set_valid_targets(self, new_targets: th.Tensor) -> bool:
        """
        Set new targets only if they are valid
        
        Args:
            new_targets: Tensor of shape (num_envs, 3) with new target positions
            
        Returns:
            Boolean indicating if all targets are valid
        """
        valid_targets = self.validate_target_position(new_targets)
        
        if valid_targets.all():
            self.target = new_targets.to(self.device)
            return True
        else:
            print(f"Warning: {valid_targets.sum().item()} out of {len(valid_targets)} targets are invalid")
            return False

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        dis_scale = (self.target - self.position).norm(dim=1, keepdim=True).detach().clamp_min(self.max_sense_radius)
        state = th.hstack([
            (self.target - self.position) / 10,
            self.orientation,
            self.velocity / 10,
            self.angular_velocity / 10,
        ]).to(self.device)

        # Add depth sensor observation if visual is enabled and depth data is available
        if self.visual and "depth" in self.sensor_obs:
            depth = th.from_numpy(self.sensor_obs["depth"]).to(self.device)
            # Normalize depth to [0, 1] range
            depth = (depth / 10).clamp(max=1)
        else:
            # Provide zero depth if no sensor data available
            depth = th.zeros((self.num_envs, 1, 64, 64), device=self.device)

        return TensorDict({
            "state": state,
            "depth": depth,
            "target": self.target.to(self.device),
        })





