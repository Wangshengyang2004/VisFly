import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional
import os, sys
import cv2

from VisFly.utils.evaluate import TestBase
from VisFly.envs.HoverEnv import HoverEnv2
# from VisFly.utils.FigFashion.FigFashion import FigFon  # Uncomment if you use custom figure styles

class Test(TestBase):
    def __init__(self, env, model, name, save_path: Optional[str] = None):
        super(Test, self).__init__(env=env, model=model, name=name, save_path=save_path)

    def create_combined_video_frame(self, global_frame, agent_sensor_data, timestep_idx):
        global_h, global_w = global_frame.shape[:2]
        agent_frames = []
        if "depth" in agent_sensor_data:
            depth_data = agent_sensor_data["depth"]
            if hasattr(depth_data, 'cpu'):
                depth_data = depth_data.cpu().numpy()
            elif hasattr(depth_data, 'numpy'):
                depth_data = depth_data.numpy()
            for agent_idx in range(depth_data.shape[0]):
                depth_frame = depth_data[agent_idx, 0]
                depth_max = depth_frame.max()
                if depth_max > 0:
                    depth_normalized = ((depth_frame / depth_max) * 255).astype(np.uint8)
                else:
                    depth_normalized = np.zeros_like(depth_frame, dtype=np.uint8)
                depth_rgb = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)
                agent_frames.append(cv2.resize(depth_rgb, (200, 200)))
        if agent_frames:
            num_agents = len(agent_frames)
            grid_cols = min(4, num_agents)
            grid_rows = (num_agents + grid_cols - 1) // grid_cols
            agent_grid_h = grid_rows * 200
            agent_grid_w = grid_cols * 200
            agent_grid = np.zeros((agent_grid_h, agent_grid_w, 3), dtype=np.uint8)
            for i, frame in enumerate(agent_frames):
                row = i // grid_cols
                col = i % grid_cols
                y_start = row * 200
                x_start = col * 200
                agent_grid[y_start:y_start+200, x_start:x_start+200] = frame
            global_resized = cv2.resize(global_frame, (int(global_w * agent_grid_h / global_h), agent_grid_h))
            combined_frame = np.hstack([global_resized, agent_grid])
        else:
            combined_frame = global_frame
        cv2.putText(combined_frame, f"Timestep: {timestep_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return combined_frame

    def save_combined_video(self, episode_dir):
        if not self.render_image_all:
            print("No render images available for video creation")
            return
        try:
            print(f"Creating combined video for episode...")
            combined_frames = []
            for i, (render_frame, obs_data) in enumerate(zip(self.render_image_all, self.obs_all)):
                render_bgr = cv2.cvtColor(render_frame, cv2.COLOR_RGB2BGR)
                combined_frame = self.create_combined_video_frame(render_bgr, obs_data, i)
                combined_frames.append(combined_frame)
            if not combined_frames:
                print("No frames to save")
                return
            height, width = combined_frames[0].shape[:2]
            fps = int(1.0 / self.env.envs.dynamics.dt)
            video_path = os.path.join(episode_dir, "combined_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            for frame in combined_frames:
                video_writer.write(frame)
            video_writer.release()
            print(f"Combined video saved: {video_path}")
            self.save_individual_videos(episode_dir)
        except Exception as e:
            print(f"Error creating combined video: {e}")
            print("Continuing without video generation...")

    def save_individual_videos(self, episode_dir):
        if not self.render_image_all:
            return
        try:
            self.save_global_video(episode_dir)
            self.save_agent_depth_videos(episode_dir)
        except Exception as e:
            print(f"Error saving individual videos: {e}")
            print("Continuing without individual video generation...")

    def save_global_video(self, episode_dir):
        try:
            height, width, _ = self.render_image_all[0].shape
            fps = int(1.0 / self.env.envs.dynamics.dt)
            video_path = os.path.join(episode_dir, "global_camera.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            for frame in self.render_image_all:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
            video_writer.release()
            print(f"Global camera video saved: {video_path}")
        except Exception as e:
            print(f"Error saving global video: {e}")
            print("Continuing without global video...")

    def save_agent_depth_videos(self, episode_dir):
        if not self.obs_all or "depth" not in self.obs_all[0]:
            return
        try:
            num_agents = self.obs_all[0]["depth"].shape[0]
            fps = int(1.0 / self.env.envs.dynamics.dt)
            agent_writers = {}
            for agent_idx in range(num_agents):
                video_path = os.path.join(episode_dir, f"agent_{agent_idx}_depth.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                agent_writers[agent_idx] = cv2.VideoWriter(video_path, fourcc, fps, (200, 200))
            for obs_data in self.obs_all:
                if "depth" in obs_data:
                    depth_data = obs_data["depth"]
                    for agent_idx in range(num_agents):
                        if agent_idx < depth_data.shape[0]:
                            depth_frame = depth_data[agent_idx, 0]
                            if hasattr(depth_frame, 'cpu'):
                                depth_frame = depth_frame.cpu().numpy()
                            elif hasattr(depth_frame, 'numpy'):
                                depth_frame = depth_frame.numpy()
                            depth_normalized = ((depth_frame / (depth_frame.max() + 1e-6)) * 255).astype(np.uint8)
                            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)
                            depth_resized = cv2.resize(depth_colored, (200, 200))
                            agent_writers[agent_idx].write(depth_resized)
            for agent_idx, writer in agent_writers.items():
                writer.release()
                print(f"Agent {agent_idx} depth video saved: agent_{agent_idx}_depth.mp4")
        except Exception as e:
            print(f"Error saving agent depth videos: {e}")
            print("Continuing without agent depth videos...")

    def draw(self, names=None):
        state_data = [state for state in self.state_all]
        state_data = [state.cpu() if hasattr(state, 'cpu') else state for state in state_data]
        state_data = np.array(state_data)
        absolute_positions = state_data[:, :, 0:3]
        targets = self.obs_all[0]["target"]
        if hasattr(targets, 'cpu'):
            targets = targets.cpu().numpy()
        num_envs = state_data.shape[1]
        
        # Fix time array construction to handle single timestep case
        t_list = []
        for time_tensor in self.t:
            if hasattr(time_tensor, 'cpu'):
                t_val = time_tensor.cpu().numpy()
            else:
                t_val = np.array(time_tensor)
            
            # Handle different shapes of time data
            if t_val.ndim == 0:  # scalar
                t_list.append(float(t_val))
            elif t_val.ndim == 1 and len(t_val) > 0:  # 1D array
                t_list.append(float(t_val[0]))
            else:
                t_list.append(0.0)  # fallback
        
        t = np.array(t_list)
        
        # Ensure t has the same length as state_data
        if len(t) != state_data.shape[0]:
            print(f"Warning: Time array length {len(t)} doesn't match data length {state_data.shape[0]}")
            # Create a proper time array based on dt
            dt = 0.02  # default dt
            if hasattr(self.env, 'envs') and hasattr(self.env.envs, 'dynamics'):
                dt = self.env.envs.dynamics.dt
            t = np.arange(state_data.shape[0]) * dt
        
        fig = plt.figure(figsize=(16, 12))
        def _plot_with_mean(ax, series, labels, title, ylabel, fill_alpha=0.15):
            if series.ndim == 2:
                series = series[:, None, :]
            mean_vals = series.mean(axis=1)
            std_vals = series.std(axis=1)
            
            # Ensure t and mean_vals have compatible dimensions
            if len(t) != mean_vals.shape[0]:
                print(f"Warning: Time array length {len(t)} doesn't match mean_vals length {mean_vals.shape[0]}")
                return  # Skip plotting if dimensions don't match
            
            for ch in range(series.shape[2]):
                ax.plot(t, mean_vals[:, ch], linewidth=2, label=labels[ch])
                ax.fill_between(
                    t,
                    mean_vals[:, ch] - std_vals[:, ch],
                    mean_vals[:, ch] + std_vals[:, ch],
                    alpha=fill_alpha,
                )
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Time (s)")
            ax.legend()
            ax.grid(True)
        
        ax1 = plt.subplot(2, 2, 1)
        _plot_with_mean(ax1, absolute_positions, ["x", "y", "z"], "Position vs Time", "Position (m)")
        
        ax2 = plt.subplot(2, 2, 2, projection='3d')
        cmap = plt.cm.get_cmap('tab20', num_envs)
        for agent_idx in range(num_envs):
            x_pos = absolute_positions[:, agent_idx, 0]
            y_pos = absolute_positions[:, agent_idx, 1]
            z_pos = absolute_positions[:, agent_idx, 2]
            vel = state_data[:, agent_idx, 7:10]
            speed = np.linalg.norm(vel, axis=1)
            ax2.plot(x_pos, y_pos, z_pos, color='lightgrey', linewidth=0.7, alpha=0.6)
            sc = ax2.scatter(x_pos, y_pos, z_pos, c=speed, cmap='viridis', s=8, alpha=0.9)
            ax2.scatter(x_pos[0], y_pos[0], z_pos[0], color='green', s=50, marker='o', edgecolor='black', linewidth=1)
            ax2.scatter(x_pos[-1], y_pos[-1], z_pos[-1], color='blue', s=50, marker='s', edgecolor='black', linewidth=1)
        
        if "target" in self.obs_all[0]:
            for agent_idx in range(num_envs):
                if len(targets.shape) == 1:
                    if agent_idx == 0 and len(targets) >= 3:
                        ax2.scatter(targets[0], targets[1], targets[2], color='gold', s=80, marker='*', edgecolor='black', linewidth=1, label='Target')
                elif len(targets.shape) == 2:
                    if targets.shape[0] > agent_idx:
                        target_pos = targets[agent_idx]
                        if len(target_pos) >= 3:
                            ax2.scatter(target_pos[0], target_pos[1], target_pos[2], color='gold', s=80, marker='*', edgecolor='black', linewidth=1, label='Target' if agent_idx == 0 else None)
        
        ax2.set_xlabel("X Position (m)")
        ax2.set_ylabel("Y Position (m)")
        ax2.set_zlabel("Z Position (m)")
        ax2.set_title("3D Trajectory")
        
        all_x = absolute_positions[:, :, 0].flatten()
        all_y = absolute_positions[:, :, 1].flatten()
        all_z = absolute_positions[:, :, 2].flatten()
        
        if "target" in self.obs_all[0]:
            if len(targets.shape) == 1 and len(targets) >= 3:
                all_x = np.append(all_x, targets[0])
                all_y = np.append(all_y, targets[1])
                all_z = np.append(all_z, targets[2])
            elif len(targets.shape) == 2:
                for i in range(min(targets.shape[0], num_envs)):
                    if len(targets[i]) >= 3:
                        all_x = np.append(all_x, targets[i][0])
                        all_y = np.append(all_y, targets[i][1])
                        all_z = np.append(all_z, targets[i][2])
        
        max_range = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max() / 2.0
        mid_x = (all_x.max()+all_x.min()) * 0.5
        mid_y = (all_y.max()+all_y.min()) * 0.5
        mid_z = (all_z.max()+all_z.min()) * 0.5
        ax2.set_xlim(mid_x - max_range, mid_x + max_range)
        ax2.set_ylim(mid_y - max_range, mid_y + max_range)
        ax2.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Convert quaternions to Euler angles (XYZ) - simple implementation
        euler_angles = np.zeros((state_data.shape[0], state_data.shape[1], 3))
        for t_idx in range(state_data.shape[0]):
            for agent in range(state_data.shape[1]):
                qw, qx, qy, qz = state_data[t_idx, agent, 3:7]  # w, x, y, z
                
                # Roll (x-axis rotation)
                sinr_cosp = 2 * (qw * qx + qy * qz)
                cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
                roll = np.arctan2(sinr_cosp, cosr_cosp)
                
                # Pitch (y-axis rotation)
                sinp = 2 * (qw * qy - qz * qx)
                if abs(sinp) >= 1:
                    pitch = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
                else:
                    pitch = np.arcsin(sinp)
                
                # Yaw (z-axis rotation)
                siny_cosp = 2 * (qw * qz + qx * qy)
                cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
                yaw = np.arctan2(siny_cosp, cosy_cosp)
                
                # Convert to degrees
                euler_angles[t_idx, agent, 0] = np.degrees(roll)   # roll
                euler_angles[t_idx, agent, 1] = np.degrees(pitch)  # pitch
                euler_angles[t_idx, agent, 2] = np.degrees(yaw)    # yaw
        
        ax3 = plt.subplot(2, 2, 3)
        _plot_with_mean(ax3, euler_angles, ["roll", "pitch", "yaw"], "Orientation Euler (XYZ)", "Euler Angles (degrees)")
        ax4 = plt.subplot(2, 2, 4)
        _plot_with_mean(ax4, state_data[:, :, 10:13], ["wx", "wy", "wz"], "Angular Velocity", "Angular Velocity (rad/s)")
        
        plt.tight_layout()
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        fig.savefig(f"{self.save_path}/trajectory_plot.png", dpi=150, bbox_inches='tight')
        print(f"Trajectory plot saved to: {self.save_path}/trajectory_plot.png")
        try:
            plt.show()
        except Exception as e:
            print(f"Could not display plot (remote server without display): {e}")
            print("Plot has been saved to file instead.")
        return [fig]

# Usage example (customize as needed):
# from VisFly.envs.HoverEnv import HoverEnv2
# from VisFly.utils.algorithms.BPTT import BPTT
# env = HoverEnv2(...)
# model = BPTT(...)
# test = Test(env, model, name="hover_test", save_path="./test_results")
# test.run()  # or whatever method you use to execute the test 