import numpy as np
from VisFly.utils.evaluate import TestBase
import os, sys
from typing import Optional
from matplotlib import pyplot as plt
from VisFly.utils.FigFashion.FigFashion import FigFon
import torch


class Test(TestBase):
    def __init__(self,
                 model,
                 name,
                 save_path: Optional[str] = None,
                 env=None
                 ):
        # Initialize base with model and environment
        super(Test, self).__init__(env=env, model=model, name=name, save_path=save_path)

    def draw(self, names=None):
        # Collect state trajectories
        state_data = [obs["state"] for obs in self.obs_all]
        state_data = np.array(state_data)  # shape: (T, N, D)
        # Convert all elements in self.t to numpy or float
        def to_numpy_or_float(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy() if x.numel() > 1 else float(x)
            return x

        t_list = [to_numpy_or_float(x) for x in self.t]
        # If t_list is a list of arrays/lists, concatenate; else, flatten
        if any(isinstance(i, (list, np.ndarray)) for i in t_list):
            t = np.concatenate(t_list)
        else:
            t = np.array(t_list).flatten()
        # Use indices as time to ensure t matches state_data length
        t = np.arange(len(state_data))
        print("t shape:", t.shape)
        print("state_data shape:", state_data.shape)
        figs = []
        # Plot per-agent state components
        for i in range(self.model.env.num_envs):
            fig = plt.figure(figsize=(6, 5))
            # Relative position
            plt.subplot(2, 2, 1)
            for j, label in enumerate(["dx", "dy", "dz"]):
                plt.plot(t, state_data[:, i, j], label=label)
            plt.legend()
            # Orientation quaternion
            plt.subplot(2, 2, 2)
            for j, label in enumerate(["qw", "qx", "qy", "qz"]):
                plt.plot(t, state_data[:, i, 3+j], label=label)
            plt.legend()
            # Linear velocity
            plt.subplot(2, 2, 3)
            for j, label in enumerate(["vx", "vy", "vz"]):
                plt.plot(t, state_data[:, i, 7+j], label=label)
            plt.legend()
            # Angular velocity
            plt.subplot(2, 2, 4)
            for j, label in enumerate(["wx", "wy", "wz"]):
                plt.plot(t, state_data[:, i, 10+j], label=label)
            plt.legend()
            plt.tight_layout()
            figs.append(fig)
        return figs 