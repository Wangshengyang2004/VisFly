#!/usr/bin/env python3
"""
Unified 360-degree panoramic picture creation tool for VisFly scenes.

This script combines the best features from all existing 360 camera implementations:
- create_360_final.py: Multiple camera positions and panorama stitching
- create_360_simple.py: Rotating camera views
- create_360_working.py: Synthetic scene visualization
- create_360_picture.py: Multi-camera 360 setup
- create_360_simple_scene.py: Scene manager integration

Usage:
    python create_360_unified.py --scene <scene_path> --mode <capture_mode>

Modes:
    rotating: Camera rotates around the scene
    multi-camera: Uses 6 directional cameras
    synthetic: Creates synthetic 3D visualization
    hybrid: Combines rotating and multi-camera approaches
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch as th
from typing import List, Tuple, Dict, Optional

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    from VisFly.envs.NavigationEnv import NavigationEnv2
    from habitat_sim.sensor import SensorType
    NAVIGATION_ENV_AVAILABLE = True
except ImportError as e:
    print(f"Warning: NavigationEnv2 not available: {e}")
    NAVIGATION_ENV_AVAILABLE = False

class Unified360Capture:
    """Unified 360-degree scene capture class."""
    
    def __init__(self, scene_path: str, resolution: Tuple[int, int] = (512, 512)):
        self.scene_path = scene_path
        self.resolution = resolution
        self.env = None
        
    def create_environment(self, sensor_mode: str = "single") -> bool:
        """Create environment based on sensor mode."""
        if not NAVIGATION_ENV_AVAILABLE:
            print("NavigationEnv2 not available, using synthetic mode")
            return False
            
        try:
            # Common configuration
            random_kwargs = {
                "state_generator": {
                    "class": "Uniform",
                    "kwargs": [
                        {"position": {"mean": [0., 0., 1.5], "half": [2.0, 2.0, 1.0]}},
                    ]
                },
                "noise_kwargs": {}
            }
            
            scene_kwargs = {
                "path": self.scene_path,
                "render_settings": {
                    "mode": "fix",
                    "view": "custom",
                    "resolution": list(self.resolution),
                    "position": th.tensor([[0., 1.5, 0.]]),
                    "line_width": 6.,
                    "trajectory": False,
                }
            }
            
            dynamics_kwargs = {
                "action_type": "bodyrate",
                "ori_output_type": "quaternion",
                "dt": 0.005,
                "ctrl_dt": 0.03,
                "ctrl_delay": True,
                "comm_delay": 0.09,
                "action_space": (-1, 1),
                "integrator": "euler",
                "drag_random": 0,
            }
            
            sensor_kwargs = []
            if sensor_mode == "multi-camera":
                sensor_kwargs = [
                    {
                        "class": "RGBCamera",
                        "kwargs": {
                            "uuid": "front",
                            "resolution": list(self.resolution),
                            "position": [0, 0.2, 0],
                            "orientation": [0, 0, 0, 1],
                            "horizontal_fov": 90
                        }
                    },
                    {
                        "class": "RGBCamera", 
                        "kwargs": {
                            "uuid": "back",
                            "resolution": list(self.resolution),
                            "position": [0, 0.2, 0],
                            "orientation": [0, 1, 0, 0],
                            "horizontal_fov": 90
                        }
                    },
                    {
                        "class": "RGBCamera",
                        "kwargs": {
                            "uuid": "left",
                            "resolution": list(self.resolution),
                            "position": [0, 0.2, 0],
                            "orientation": [0, 0.707, 0, 0.707],
                            "horizontal_fov": 90
                        }
                    },
                    {
                        "class": "RGBCamera",
                        "kwargs": {
                            "uuid": "right",
                            "resolution": list(self.resolution),
                            "position": [0, 0.2, 0],
                            "orientation": [0, -0.707, 0, 0.707],
                            "horizontal_fov": 90
                        }
                    },
                    {
                        "class": "RGBCamera",
                        "kwargs": {
                            "uuid": "up",
                            "resolution": list(self.resolution),
                            "position": [0, 0.2, 0],
                            "orientation": [0.707, 0, 0, 0.707],
                            "horizontal_fov": 90
                        }
                    },
                    {
                        "class": "RGBCamera",
                        "kwargs": {
                            "uuid": "down",
                            "resolution": list(self.resolution),
                            "position": [0, 0.2, 0],
                            "orientation": [-0.707, 0, 0, 0.707],
                            "horizontal_fov": 90
                        }
                    }
                ]
            
            self.env = NavigationEnv2(
                visual=True,
                num_scene=1,
                num_agent_per_scene=1,
                random_kwargs=random_kwargs,
                scene_kwargs=scene_kwargs,
                dynamics_kwargs=dynamics_kwargs,
                sensor_kwargs=sensor_kwargs,
            )
            
            self.env.reset()
            print("Environment created successfully!")
            return True
            
        except Exception as e:
            print(f"Failed to create environment: {e}")
            return False
    
    def capture_rotating_views(self, num_views: int = 8, radius: float = 5.0, 
                              height: float = 2.0) -> List[np.ndarray]:
        """Capture rotating camera views around the scene."""
        if not self.env:
            raise RuntimeError("Environment not available for rotating views")
            
        images = []
        scene_manager = self.env.envs.sceneManager
        
        for i in range(num_views):
            angle = 2 * np.pi * i / num_views
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            
            camera_pos = np.array([x, height, z], dtype=np.float32)
            look_at = np.array([0., 1.0, 0.], dtype=np.float32)
            
            try:
                # Update render settings with new camera position and lookat
                # SceneManager expects position to be [camera_pos, lookat_pos]
                new_position = th.stack([
                    th.tensor(camera_pos, dtype=th.float32),
                    th.tensor(look_at, dtype=th.float32)
                ])  # Shape: (2, 3)
                
                # Update the render settings
                scene_manager.render_settings["position"] = new_position
                
                # Render the scene
                img = self.env.render(is_draw_axes=False)
                if img is not None and len(img) > 0:
                    images.append(img[0])
                    print(f"Successfully captured rotating view {i+1}/{num_views}")
                else:
                    raise RuntimeError(f"No image rendered for view {i}")
                    
            except Exception as e:
                raise RuntimeError(f"Failed to capture rotating view {i}: {e}")
        
        return images
    
    def capture_multi_camera_views(self) -> Dict[str, np.ndarray]:
        """Capture views from multiple directional cameras."""
        if not self.env or not hasattr(self.env, 'sensor_obs'):
            raise RuntimeError("Environment not available for multi-camera views")
            
        try:
            obs = self.env.sensor_obs
            views = {}
            
            required_directions = ["front", "back", "left", "right", "up", "down"]
            for direction in required_directions:
                if direction in obs:
                    img = obs[direction][0]
                    # Handle both tensor and numpy array cases
                    if hasattr(img, 'cpu'):
                        img = img.cpu().numpy()
                    elif hasattr(img, 'numpy'):
                        img = img.numpy()
                    # img is already a numpy array
                    views[direction] = img
                else:
                    raise KeyError(f"Missing {direction} camera in sensor observations")
                    
            if len(views) != len(required_directions):
                raise RuntimeError("Not all directional cameras available")
                
            return views
            
        except Exception as e:
            raise RuntimeError(f"Failed to capture multi-camera views: {e}")
    
    def create_synthetic_rotating_views(self, num_views: int = 8, radius: float = 5.0, 
                                       height: float = 2.0) -> List[np.ndarray]:
        """Create synthetic rotating views when environment is not available."""
        images = []
        
        for i in range(num_views):
            angle = 2 * np.pi * i / num_views
            view_name = f"view_{i}"
            
            img = self.create_synthetic_scene_view(view_name, angle)
            images.append(img)
            
        return images
    
    def create_synthetic_multi_camera_views(self) -> Dict[str, np.ndarray]:
        """Create synthetic multi-camera views."""
        views = {}
        directions = ["front", "back", "left", "right", "up", "down"]
        
        for direction in directions:
            views[direction] = self.create_synthetic_scene_view(direction)
            
        return views
    
    def create_synthetic_scene_view(self, view_name: str, angle: float = 0) -> np.ndarray:
        """Create a synthetic 3D scene visualization."""
        width, height = self.resolution
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Background gradient (sky-like)
        for y in range(height):
            color_val = max(0, min(255, int(200 - (y / height) * 100)))
            img[y, :] = [color_val, min(255, color_val + 50), min(255, color_val + 100)]
        
        # Draw box scene elements
        center_x, center_y = width // 2, height // 2
        
        # Ground plane
        ground_y = int(height * 0.7)
        img[ground_y:, :] = [100, 100, 50]  # Brownish ground
        
        # Box structure based on view
        wall_color = (150, 150, 150)
        
        if "front" in view_name or view_name.startswith("view_"):
            # Front view shows rectangular box
            box_width, box_height = 200, 150
            if view_name.startswith("view_"):
                # Vary size based on angle
                scale = 0.8 + 0.4 * np.cos(float(view_name.split('_')[1]) * 0.5)
                box_width = int(200 * scale)
                box_height = int(150 * scale)
                
            top_left = (center_x - box_width//2, center_y - box_height//2)
            bottom_right = (center_x + box_width//2, center_y + box_height//2)
            cv2.rectangle(img, top_left, bottom_right, wall_color, 2)
            cv2.rectangle(img, top_left, bottom_right, (200, 200, 200), -1)
            
        elif "back" in view_name:
            # Back view
            box_width, box_height = 180, 140
            top_left = (center_x - box_width//2, center_y - box_height//2)
            bottom_right = (center_x + box_width//2, center_y + box_height//2)
            cv2.rectangle(img, top_left, bottom_right, wall_color, 2)
            
        elif "left" in view_name or "right" in view_name:
            # Side views
            box_width, box_height = 120, 150
            top_left = (center_x - box_width//2, center_y - box_height//2)
            bottom_right = (center_x + box_width//2, center_y + box_height//2)
            cv2.rectangle(img, top_left, bottom_right, wall_color, 2)
            
        elif "up" in view_name:
            # Top view
            box_width, box_height = 150, 150
            top_left = (center_x - box_width//2, center_y - box_height//2)
            bottom_right = (center_x + box_width//2, center_y + box_height//2)
            cv2.rectangle(img, top_left, bottom_right, (100, 100, 100), -1)
            cv2.rectangle(img, top_left, bottom_right, wall_color, 2)
            
        elif "down" in view_name:
            # Bottom view
            box_width, box_height = 150, 150
            top_left = (center_x - box_width//2, center_y - box_height//2)
            bottom_right = (center_x + box_width//2, center_y + box_height//2)
            cv2.rectangle(img, top_left, bottom_right, (120, 120, 120), -1)
            cv2.rectangle(img, top_left, bottom_right, wall_color, 2)
        
        # Add view label
        cv2.putText(img, view_name.upper(), (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return img
    
    def create_fallback_view(self, view_name: str) -> np.ndarray:
        """Create a simple fallback visualization."""
        width, height = self.resolution
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create gradient background
        for y in range(height):
            img[y, :] = [50 + (y // 4), 100 + (y // 8), 150 + (y // 16)]
        
        # Draw simple box
        center_x, center_y = width // 2, height // 2
        cv2.rectangle(img, (100, 200), (400, 400), (200, 200, 200), 3)
        cv2.rectangle(img, (120, 220), (380, 380), (150, 150, 150), -1)
        
        # Add label
        cv2.putText(img, view_name.upper(), (200, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return img
    
    def create_panorama(self, images: List[np.ndarray], layout: str = "grid") -> np.ndarray:
        """Create panorama from multiple images with proper stitching."""
        if not images:
            return np.zeros((512, 1024, 3), dtype=np.uint8)
        
        if layout == "grid":
            # Create 2x4 grid for 8 images
            target_size = (256, 256)
            resized_images = [cv2.resize(img, target_size) for img in images]
            
            if len(resized_images) >= 8:
                top_row = np.hstack(resized_images[:4])
                bottom_row = np.hstack(resized_images[4:8])
                panorama = np.vstack([top_row, bottom_row])
            else:
                # Single row
                panorama = np.hstack(resized_images)
                
        elif layout == "horizontal":
            # Simple horizontal concatenation
            target_height = 512
            scale = target_height / images[0].shape[0]
            target_width = int(images[0].shape[1] * scale)
            
            resized_images = [cv2.resize(img, (target_width, target_height)) for img in images]
            panorama = np.hstack(resized_images)
            
        elif layout == "cylindrical":
            # Create proper 360 equirectangular panorama
            panorama = self.create_equirectangular_from_rotating_views(images)
            
        return panorama
    
    def create_equirectangular_from_rotating_views(self, images: List[np.ndarray], 
                                                  pano_width: int = 2048, 
                                                  pano_height: int = 1024) -> np.ndarray:
        """Create equirectangular panorama from rotating camera views with seamless stitching."""
        if not images:
            return np.zeros((pano_height, pano_width, 3), dtype=np.uint8)
        
        # Use faster simple blending for now, can be optimized later
        return self.create_seamless_cylindrical_panorama(images, pano_width, pano_height)
    
    def create_seamless_cylindrical_panorama(self, images: List[np.ndarray], 
                                           pano_width: int = 2048, 
                                           pano_height: int = 1024) -> np.ndarray:
        """Create seamless cylindrical panorama with proper blending at boundaries."""
        if not images:
            return np.zeros((pano_height, pano_width, 3), dtype=np.uint8)
        
        num_views = len(images)
        if num_views == 0:
            return np.zeros((pano_height, pano_width, 3), dtype=np.uint8)
        
        # Resize all images to same height
        target_height = pano_height
        resized_images = []
        for img in images:
            aspect_ratio = img.shape[1] / img.shape[0]
            target_width = int(target_height * aspect_ratio)
            resized = cv2.resize(img, (target_width, target_height))
            resized_images.append(resized)
        
        # Calculate width per view in final panorama
        view_width = pano_width // num_views
        overlap_width = max(view_width // 10, 20)  # 10% overlap or minimum 20px
        
        # Create panorama with blending
        panorama = np.zeros((pano_height, pano_width, 3), dtype=np.float32)
        
        for i, img in enumerate(resized_images):
            img_float = img.astype(np.float32)
            img_h, img_w = img.shape[:2]
            
            # Calculate position in panorama
            start_x = i * view_width
            end_x = min((i + 1) * view_width, pano_width)
            actual_width = end_x - start_x
            
            # Resize image to fit assigned width
            if actual_width > 0:
                view_resized = cv2.resize(img_float, (actual_width, target_height))
                
                # Create alpha blending mask for seamless edges
                alpha_mask = np.ones((target_height, actual_width), dtype=np.float32)
                
                # Apply feathering at edges for blending
                if i > 0:  # Left edge blending
                    fade_width = min(overlap_width, actual_width // 2)
                    for x in range(fade_width):
                        alpha_mask[:, x] = x / fade_width
                
                if i < num_views - 1:  # Right edge blending
                    fade_width = min(overlap_width, actual_width // 2)
                    for x in range(fade_width):
                        alpha_mask[:, actual_width - 1 - x] = x / fade_width
                
                # Apply alpha blending
                for c in range(3):  # RGB channels
                    existing = panorama[:, start_x:end_x, c]
                    new_data = view_resized[:, :, c]
                    existing_alpha = (existing > 0).astype(np.float32)
                    
                    # Simple replacement with alpha blending at edges
                    panorama[:, start_x:end_x, c] = (
                        existing * (1 - alpha_mask) + 
                        new_data * alpha_mask
                    )
        
        # Handle wraparound blending (connect last image to first for seamless 360)
        if num_views > 2:
            wrap_width = min(overlap_width, view_width // 4)
            
            # Blend right edge of last image with left edge of first image
            left_edge = panorama[:, :wrap_width].copy()
            right_edge = panorama[:, -wrap_width:].copy()
            
            # Create blending weights
            blend_weights = np.linspace(0, 1, wrap_width)
            
            for c in range(3):
                for x in range(wrap_width):
                    weight = blend_weights[x]
                    # Blend left edge
                    panorama[:, x, c] = (left_edge[:, x, c] * (1 - weight) + 
                                       right_edge[:, x, c] * weight)
                    # Blend right edge
                    panorama[:, -wrap_width + x, c] = (right_edge[:, x, c] * (1 - weight) + 
                                                     left_edge[:, x, c] * weight)
        
        return panorama.astype(np.uint8)
    
    def capture_walkthrough_video(self, duration: float = 10.0, fps: int = 30, 
                                 camera_path: str = "circular") -> str:
        """Capture a walkthrough video with camera moving around the scene."""
        if not self.env:
            raise RuntimeError("Environment not available for video capture")
        
        total_frames = int(duration * fps)
        scene_manager = self.env.envs.sceneManager
        
        # Video setup
        width, height = self.resolution
        fourcc = cv2.VideoWriter_fourcc(*'H264')  # H.264 codec for best compatibility
        video_path = f"360_walkthrough_{camera_path}.mp4"
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        print(f"Recording {duration}s video at {fps} FPS ({total_frames} frames)")
        
        try:
            for frame_idx in range(total_frames):
                # Calculate camera position based on path type
                t = frame_idx / total_frames  # Normalized time [0, 1]
                camera_pos, look_at = self.calculate_camera_position(t, camera_path)
                
                # Update camera position
                new_position = th.stack([
                    th.tensor(camera_pos, dtype=th.float32),
                    th.tensor(look_at, dtype=th.float32)
                ])
                scene_manager.render_settings["position"] = new_position
                
                # Render frame
                img = self.env.render(is_draw_axes=False)
                if img is not None and len(img) > 0:
                    frame_bgr = cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR)
                    video_writer.write(frame_bgr)
                    
                    # Progress indicator
                    if (frame_idx + 1) % (total_frames // 10) == 0:
                        progress = (frame_idx + 1) / total_frames * 100
                        print(f"Progress: {progress:.0f}%")
                else:
                    raise RuntimeError(f"Failed to render frame {frame_idx}")
        
        except Exception as e:
            print(f"Video capture failed: {e}")
            video_writer.release()
            return None
        
        video_writer.release()
        print(f"Video saved: {video_path}")
        return video_path
    
    def calculate_camera_position(self, t: float, path_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate camera position and look-at point based on path type and time."""
        scene_center = np.array([0.0, 1.5, 0.0])  # Center of scene
        
        if path_type == "circular":
            # Camera moves in a circle around the scene at constant height
            radius = 6.0
            height = 2.5
            angle = t * 2 * np.pi
            
            camera_pos = np.array([
                radius * np.cos(angle),
                height,
                radius * np.sin(angle)
            ])
            look_at = scene_center
            
        elif path_type == "orbit":
            # Camera orbits while also moving up and down
            radius = 5.0
            height_base = 1.5
            height_variation = 2.0
            angle = t * 2 * np.pi
            
            camera_pos = np.array([
                radius * np.cos(angle),
                height_base + height_variation * np.sin(angle * 2),
                radius * np.sin(angle)
            ])
            look_at = scene_center
            
        elif path_type == "spiral":
            # Camera moves in a spiral, getting closer and higher
            radius_start = 8.0
            radius_end = 3.0
            height_start = 1.0
            height_end = 4.0
            angle = t * 4 * np.pi  # Two full rotations
            
            radius = radius_start + (radius_end - radius_start) * t
            height = height_start + (height_end - height_start) * t
            
            camera_pos = np.array([
                radius * np.cos(angle),
                height,
                radius * np.sin(angle)
            ])
            look_at = scene_center
            
        elif path_type == "flythrough":
            # Camera flies through the scene on a curved path
            if t < 0.33:
                # Approach from distance
                progress = t / 0.33
                camera_pos = np.array([
                    10.0 - 8.0 * progress,
                    3.0 - 1.0 * progress,
                    5.0 - 3.0 * progress
                ])
            elif t < 0.67:
                # Circle around close to scene
                progress = (t - 0.33) / 0.34
                angle = progress * np.pi
                radius = 2.5
                camera_pos = np.array([
                    2.0 + radius * np.cos(angle),
                    2.0 + 0.5 * np.sin(angle * 2),
                    2.0 + radius * np.sin(angle)
                ])
            else:
                # Fly away
                progress = (t - 0.67) / 0.33
                camera_pos = np.array([
                    -2.0 - 8.0 * progress,
                    2.5 + 2.0 * progress,
                    -2.0 - 3.0 * progress
                ])
            
            look_at = scene_center
        
        else:
            # Default circular path
            radius = 5.0
            height = 2.0
            angle = t * 2 * np.pi
            camera_pos = np.array([
                radius * np.cos(angle),
                height,
                radius * np.sin(angle)
            ])
            look_at = scene_center
        
        return camera_pos, look_at
    
    def create_equirectangular(self, views: Dict[str, np.ndarray]) -> np.ndarray:
        """Create equirectangular projection from directional views."""
        if not views:
            return np.zeros((512, 1024, 3), dtype=np.uint8)
        
        height = 512
        width = 1024
        panorama = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Standard layout for 6 directional views
        directions = ["front", "right", "back", "left", "up", "down"]
        
        # Resize all views to same size
        target_size = (256, 256)
        resized_views = {}
        
        for direction in directions:
            if direction in views:
                resized_views[direction] = cv2.resize(views[direction], target_size)
            else:
                resized_views[direction] = np.zeros(target_size + (3,), dtype=np.uint8)
        
        # Place views in panorama
        # Top row: up, front, right, back
        if "up" in resized_views:
            panorama[0:256, 0:256] = resized_views["up"]
        if "front" in resized_views:
            panorama[0:256, 256:512] = resized_views["front"]
        if "right" in resized_views:
            panorama[0:256, 512:768] = resized_views["right"]
        if "back" in resized_views:
            panorama[0:256, 768:1024] = resized_views["back"]
            
        # Bottom row: down, left, empty, empty
        if "down" in resized_views:
            panorama[256:512, 0:256] = resized_views["down"]
        if "left" in resized_views:
            panorama[256:512, 256:512] = resized_views["left"]
        
        return panorama
    
    def save_images(self, images: List[np.ndarray], prefix: str = "view"):
        """Save individual images."""
        for i, img in enumerate(images):
            filename = f"{prefix}_{i:02d}.png"
            cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"Saved {filename}")
    
    def cleanup(self):
        """Clean up environment resources."""
        if self.env:
            try:
                self.env.close()
            except:
                pass
    
    def capture_360_scene(self, mode: str = "rotating", 
                         save_individual: bool = True,
                         num_views: int = 8,
                         pano_size: Tuple[int, int] = (2048, 1024),
                         video_duration: float = 10.0,
                         video_fps: int = 30,
                         camera_path: str = "circular") -> Dict[str, np.ndarray]:
        """Main method to capture 360-degree scene."""
        print(f"Starting 360 scene capture in {mode} mode...")
        
        results = {}
        
        try:
            if mode == "video":
                # Video walkthrough mode - requires real environment
                success = self.create_environment("single")
                if not success:
                    raise RuntimeError("Real environment required for video capture")
                
                video_path = self.capture_walkthrough_video(video_duration, video_fps, camera_path)
                if video_path:
                    results["video_path"] = video_path
                    print(f"Walkthrough video created: {video_path}")
                else:
                    raise RuntimeError("Failed to create walkthrough video")
                    
            elif mode == "rotating" or mode == "multi-camera":
                # Real environment modes
                camera_mode = "single" if mode == "rotating" else "multi-camera"
                success = self.create_environment(camera_mode)
                
                if not success:
                    raise RuntimeError("Failed to create real environment")
                
                if mode == "rotating":
                    images = self.capture_rotating_views(num_views=num_views)
                    if not images:
                        raise RuntimeError("Failed to capture rotating views")
                        
                    panorama = self.create_equirectangular_from_rotating_views(images, pano_size[0], pano_size[1])
                    results["panorama"] = panorama
                    results["images"] = images
                    
                    if save_individual:
                        self.save_images(images, "rotating")
                        cv2.imwrite("360_panorama_rotating.png", 
                                  cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR))
                        
                else:  # multi-camera
                    views = self.capture_multi_camera_views()
                    if not views:
                        raise RuntimeError("Failed to capture multi-camera views")
                        
                    panorama = self.create_equirectangular(views)
                    results["panorama"] = panorama
                    results["views"] = views
                    
                    if save_individual:
                        for name, img in views.items():
                            filename = f"{name}_view.png"
                            cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                            print(f"Saved {filename}")
                        
                        cv2.imwrite("360_panorama_multicam.png", 
                                  cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR))
                        
            elif mode == "synthetic":
                # Synthetic visualization - always works
                images = self.create_synthetic_rotating_views(num_views=num_views)
                panorama = self.create_equirectangular_from_rotating_views(images, pano_size[0], pano_size[1])
                
                results["panorama"] = panorama
                results["images"] = images
                
                if save_individual:
                    self.save_images(images, "synthetic")
                    cv2.imwrite("360_panorama_synthetic.png", 
                              cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR))
                    
            elif mode == "hybrid":
                # Try real first, then synthetic with clear messaging
                try:
                    success = self.create_environment("multi-camera")
                    if success:
                        views = self.capture_multi_camera_views()
                        if views:
                            panorama = self.create_equirectangular(views)
                            results["panorama"] = panorama
                            results["views"] = views
                            
                            if save_individual:
                                cv2.imwrite("360_panorama_real.png", 
                                          cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR))
                            print("Used real environment capture")
                        else:
                            raise RuntimeError("Real capture produced no views")
                    else:
                        raise RuntimeError("Real environment unavailable")
                        
                except Exception as e:
                    print(f"Real environment failed: {e}")
                    print("Falling back to synthetic mode...")
                    
                    images = self.create_synthetic_rotating_views(num_views=num_views)
                    panorama = self.create_equirectangular_from_rotating_views(images, pano_size[0], pano_size[1])
                    results["panorama"] = panorama
                    results["images"] = images
                    
                    if save_individual:
                        self.save_images(images, "synthetic")
                        cv2.imwrite("360_panorama_synthetic.png", 
                                  cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR))
                    print("Used synthetic fallback")
        
        except Exception as e:
            print(f"Error during capture: {e}")
            raise  # Re-raise the error instead of falling back
        
        finally:
            self.cleanup()
        
        print("360 scene capture completed!")
        return results


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Create 360-degree panoramic views")
    parser.add_argument("--scene", default="VisFly/datasets/visfly-beta/configs/scenes/box15_wall_box15_wall",
                       help="Path to scene configuration")
    parser.add_argument("--mode", choices=["rotating", "multi-camera", "synthetic", "hybrid", "video"],
                       default="hybrid", help="Capture mode")
    parser.add_argument("--resolution", type=int, nargs=2, default=[512, 512],
                       help="Image resolution (width height)")
    parser.add_argument("--no-individual", action="store_true",
                       help="Don't save individual views")
    parser.add_argument("--panorama-size", type=int, nargs=2, default=[2048, 1024],
                       help="Final panorama resolution (width height)")
    parser.add_argument("--num-views", type=int, default=8,
                       help="Number of rotating views to capture (for rotating mode)")
    parser.add_argument("--video-duration", type=float, default=10.0,
                       help="Video duration in seconds (for video mode)")
    parser.add_argument("--video-fps", type=int, default=30,
                       help="Video frame rate (for video mode)")
    parser.add_argument("--camera-path", choices=["circular", "orbit", "spiral", "flythrough"],
                       default="circular", help="Camera movement path for video mode")
    
    args = parser.parse_args()
    
    print(f"Unified 360 Capture Tool")
    print(f"Scene: {args.scene}")
    print(f"Mode: {args.mode}")
    print(f"Resolution: {args.resolution}")
    
    # Create capture instance
    capture = Unified360Capture(args.scene, tuple(args.resolution))
    
    # Update panorama size and num_views if provided
    if hasattr(capture, 'pano_width'):
        capture.pano_width = args.panorama_size[0]
        capture.pano_height = args.panorama_size[1]
    
    # Perform capture
    results = capture.capture_360_scene(
        mode=args.mode,
        save_individual=not args.no_individual,
        num_views=args.num_views,
        pano_size=tuple(args.panorama_size),
        video_duration=args.video_duration,
        video_fps=args.video_fps,
        camera_path=args.camera_path
    )
    
    # Summary
    print(f"\n=== CAPTURE SUMMARY ===")
    print(f"Generated files:")
    for key in results:
        if key == "panorama":
            print(f"- 360_panorama_{args.mode}.png")
        elif key == "images":
            print(f"- {len(results[key])} individual rotating views")
        elif key == "views":
            print(f"- {len(results[key])} directional camera views")
        elif key == "video_path":
            print(f"- {results[key]} (walkthrough video)")
    
    return results


if __name__ == "__main__":
    main()