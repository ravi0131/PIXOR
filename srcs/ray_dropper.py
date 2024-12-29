import pandas as pd
import numpy as np
from typing import Tuple

class RayDropper:
    """
    Ray dropper processes 4 consecutive frames with the same drop ratio and sphercial resolution.
    For the 5th frame, the drop ratio and spherical resoultion is chosen again
    """
    def __init__(self, input_size=1):
        self.input_size = input_size
        self.counter = 0
        self.spherical_resolution = np.random.choice([600,900,1200,1500])
        self.beam_drop_ratio = np.random.choice([1, 2, 3])
        
    def _random_beam_drop(self, points: np.ndarray) -> np.ndarray:
        """
        Randomly drop beams from the point cloud data
        
        Args:
            points (nx5 np.ndarray)
            Columns = [x, y, z, beam_index, intensity] 
        
        Returns:
            nx5 np.ndarray: Points after dropping rays
            Columns = [x, y, z, beam_index, intensity]
        """
        # Randomly select a beam drop ratio from [1, 2, 3]
        beam_drop_ratio = self.beam_drop_ratio
        # Randomly select a starting beam index
        start_index = np.random.randint(0, beam_drop_ratio)

        # Apply the ray-dropping condition
        mask = (points[:, 3] - start_index) % beam_drop_ratio == 0
        return points[mask]



    def _spherical_coordinates_conversion(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """"
        Converts 3D points from Euclidean (x, y, z) to spherical coordinates (theta, phi, radial_dist).

        Args:
            points (np.ndarray): A numpy array of shape (n, 3) or (n, 4), where each row represents a 3D point.
                                If there are 4 columns, the last column will be ignored.

        Returns:
            np.ndarray: A (m, 3) array of spherical coordinates (theta, phi, radial_dist) for valid points.
            np.ndarray: A boolean mask (n,) indicating which input points have radial distance > 0.1.

        Example:
            >>> points = np.array([[1, 1, 1], [0, 0, 0], [3, 4, 5]])
            >>> spherical_coords, valid_mask = spherical_coordinates_conversion(points)
        """
        # Convert to spherical coordinates
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        radial_dist = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y, x)  # azimuth
        phi = np.arcsin(z / radial_dist)  # elevation

        # Filter out points with radial distance < 0.1
        valid_mask = radial_dist > 0.1

        return np.vstack((theta[valid_mask], phi[valid_mask], radial_dist[valid_mask])).T, valid_mask

    def _random_spherical_drop(self, points: np.ndarray) -> np.ndarray:
        """
        Randomly drop rays in the spherical coordinates
        Args:
            points (nx5 np.ndarray)
            Columns = [x, y, z, beam_index, intensity] 
        
        Returns:
            nx5 np.ndarray: Points after dropping rays
            Columns = [x, y, z, beam_index, intensity]

        Example:
            >>> points = np.array([[1, 1, 1, 0], [0, 0, 0, 1], [3, 4, 5, 2]])
            >>> points_after_spherical_drop = random_spherical_drop(points)
        """
        # Sample spherical resolutions for theta and phi
        spherical_resolution = self.spherical_resolution
        
        # Convert theta and phi to grid cells
        spherical_coords, valid_mask = self._spherical_coordinates_conversion(points)
        theta_grid = (spherical_coords[:, 0] * spherical_resolution).astype(int)
        phi_grid = (spherical_coords[:, 1] * spherical_resolution).astype(int)
        
        # Randomly sample spherical drop ratio
        spherical_drop_ratio = np.random.choice([1, 2])
        # Apply the ray-dropping condition in the spherical coordinates
        theta_mask = (theta_grid % spherical_drop_ratio == 0)
        phi_mask = (phi_grid % spherical_drop_ratio == 0)
        
        # Combine the valid_mask from earlier with spherical mask
        combined_mask = valid_mask & theta_mask & phi_mask
        return points[combined_mask]


    def drop_rays(self, points: np.ndarray) -> np.ndarray:
        """
        Randomly drop rays in the point cloud data
        Args:
            points (nx5 np.ndarray)
            Columns = [x, y, z, beam_index, intensity] 
        
        Returns:
            nx5 np.ndarray: Points after dropping rays
            Columns = [x, y, z, beam_index, intensity]
        """
        if self.counter >= self.input_size:
            self.counter = 0
            self.spherical_resolution = np.random.choice([600,900,1200,1500])
            self.beam_drop_ratio = np.random.choice([1, 2, 3])
            
        # Step 1: Random beam drop
        points_after_beam_drop = self._random_beam_drop(points)
        
        # Step 2: Spherical drop
        points_after_spherical_drop = self._random_spherical_drop(points_after_beam_drop)
        
        # print(f"ray-dropper => counter: {self.counter}, spherical_resolution: {self.spherical_resolution}, beam_drop_ratio: {self.beam_drop_ratio} ")
        self.counter += 1
        return points_after_spherical_drop
