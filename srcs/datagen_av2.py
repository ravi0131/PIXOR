import os.path
import numpy as np
import time
import torch
from utils import plot_bev, get_points_in_a_rotated_box, plot_label_map, trasform_label2metric
from torch.utils.data import Dataset, DataLoader
from av2.datasets.sensor.av2_sensor_dataloader import  AV2SensorDataLoader
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from ray_dropper import RayDropper

AV2_PATH = os.path.join(os.path.expanduser('~'), 'buni','dataset','av2-pixor')

class AV2(Dataset):

    geometry = {
        'L1': -40.0,
        'L2': 40.0,
        'W1': 0.0,
        'W2': 70.0,
        'H1': -2.5,
        'H2': 1.0,
        'input_shape': (800, 700, 36),
        'label_shape': (200, 175, 7)
    }

    target_mean = np.array([0.008, 0.001, 0.202, 0.2, 0.43, 1.368])
    target_std_dev = np.array([0.866, 0.5, 0.954, 0.668, 0.09, 0.111])


    def __init__(self,train=True):
        self.dataset_api = None
        self.train = train
        if train:
            train_path = Path(os.path.join(AV2_PATH, 'training','train'))
            self.av2_api = AV2SensorDataLoader(data_dir=train_path, labels_dir=train_path)
        else:
            test_path = Path(os.path.join(AV2_PATH, 'training','val'))
            self.av2_api = AV2SensorDataLoader(data_dir=test_path, labels_dir=test_path)
        
        self.scenes = self.av2_api.get_log_ids()
        self.global_to_scene_frame = []  # List mapping global index to (scene_id, frame_idx)
        self.total_frames = 0

    def __len__(self):
        return self.total_frames

    def __getitem__(self, item):
        scan = self.load_velo_scan(item)
        rd = RayDropper()
        scan = rd.drop_rays(scan)
        scan = self.lidar_preprocess(scan)
        scan = torch.from_numpy(scan)
        
        label_map, _ = self.get_label(item)
        self.reg_target_transform(label_map)
        label_map = torch.from_numpy(label_map)
        
        scan = scan.permute(2, 0, 1)
        label_map = label_map.permute(2, 0, 1)
        return scan, label_map, item


    def reg_target_transform(self, label_map: np.ndarray):
        '''
        Inputs are numpy arrays (not tensors!)
        :param label_map: [200 * 175 * 7] label tensor
        :return: normalized regression map for all non_zero classification locations
        '''
        cls_map = label_map[..., 0]
        reg_map = label_map[..., 1:]

        index = np.nonzero(cls_map)
        reg_map[index] = (reg_map[index] - self.target_mean)/self.target_std_dev
        
    def get_corners(self, bbox: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        return the 4 corners of the bounding box in the bird's eye view
        
        Args:
            bbox:  list of the bounding box parameters of shape (5)
        
        Returns:
            bev_corners: [4 * 2] numpy array of the 4 corners' (x, y)
            reg_target: [6] numpy array of the regression targets  
        """
        x, y, l, w, yaw = bbox
        
        bev_corners = np.zeros((4, 2), dtype=np.float32)
        # rear left
        bev_corners[0, 0] = x - l/2 * np.cos(yaw) - w/2 * np.sin(yaw)
        bev_corners[0, 1] = y - l/2 * np.sin(yaw) + w/2 * np.cos(yaw)

        # rear right
        bev_corners[1, 0] = x - l/2 * np.cos(yaw) + w/2 * np.sin(yaw)
        bev_corners[1, 1] = y - l/2 * np.sin(yaw) - w/2 * np.cos(yaw)

        # front right
        bev_corners[2, 0] = x + l/2 * np.cos(yaw) + w/2 * np.sin(yaw)
        bev_corners[2, 1] = y + l/2 * np.sin(yaw) - w/2 * np.cos(yaw)

        # front left
        bev_corners[3, 0] = x + l/2 * np.cos(yaw) - w/2 * np.sin(yaw)
        bev_corners[3, 1] = y + l/2 * np.sin(yaw) + w/2 * np.cos(yaw)

        reg_target = [np.cos(yaw), np.sin(yaw), x, y, w, l]

        return bev_corners, reg_target


    def update_label_map(self, map: np.ndarray, bev_corners: np.ndarray, reg_target: np.ndarray):
        """
        Args:
            map: [200 * 175 * 7] numpy array
            bev_corners: [4 * 2] numpy array of the 4 corners' (x, y)
            reg_target: [6] numpy array of the regression targets
        
        Returns:
            None
        """
        label_corners = (bev_corners / 4 ) / 0.1
        label_corners[:, 1] += self.geometry['label_shape'][0] / 2

        points = get_points_in_a_rotated_box(label_corners, self.geometry['label_shape'])

        for p in points:
            label_x = p[0]
            label_y = p[1]
            metric_x, metric_y = trasform_label2metric(np.array(p))
            actual_reg_target = np.copy(reg_target)
            actual_reg_target[2] = reg_target[2] - metric_x
            actual_reg_target[3] = reg_target[3] - metric_y
            actual_reg_target[4] = np.log(reg_target[4])
            actual_reg_target[5] = np.log(reg_target[5])

            map[label_y, label_x, 0] = 1.0
            map[label_y, label_x, 1:7] = actual_reg_target


    def get_label(self, index):
        '''
        :param i: the ith velodyne scan in the train/val set
        :return: label map: <--- This is the learning target
                a tensor of shape 800 * 700 * 7 representing the expected output


                label_list: <--- Intended for evaluation metrics & visualization
                a list of length n; n =  number of cars + (truck+van+tram+dontcare) in the frame
                each entry is another list, where the first element of this list indicates if the object
                is a car or one of the 'dontcare' (truck,van,etc) object

        '''
        if self.train:
            label_path = os.path.join(os.path.expanduser('~'),'buni', 'output-data','av2','bbox-estimation')
            log_id, frame_id = self.global_to_scene_frame[index]
            print(f"get_label() called => log_id is {log_id} and frame_id is {frame_id}")
            label_map = np.zeros(self.geometry['label_shape'], dtype=np.float32)
            label_list = []
        
            labels_df = pd.read_feather(os.path.join(label_path, log_id, str(frame_id) + '.feather'))
            
            for index, row in labels_df.iterrows():
                #convert row into a list
                row = row.tolist()
                corners, reg_target = self.get_corners(row)
                self.update_label_map(label_map, corners, reg_target)
                label_list.append(corners)
            return label_map, label_list
        
        else:
            log_id, frame_id = self.global_to_scene_frame[index]
            print(f"get_label() called => log_id is {log_id} and frame_id is {frame_id}")
            
            cuboids = self.av2_api.get_labels_at_lidar_timestamp(log_id, frame_id).vertices_m
            filtered_cuboids = self.filter_cuboids_by_roi(cuboids)
            labels_2d = self._extract_face_corners(filtered_cuboids, bottom_face=True) # numpy array (N, 4, 2)
            
            label_list = []
            label_map = np.zeros(self.geometry['label_shape'], dtype=np.float32)
            
            for label in labels_2d:
                l, w, angle = self.get_lwo(label)
                cx, cy = np.mean(label[:,0]), np.mean(label[:,1])
                corners, reg_target = self.get_corners([cx, cy, l, w, angle])
                self.update_label_map(label_map, corners, reg_target)
                label_list.append(corners)
            return label_map, label_list

    def get_lwo (self, corners: np.ndarray) -> tuple:
        """
        Calculates length, width and orientation bbox with clockwise ordered corners.
        
        Args:
            corners: numpy array of shape (4,2) with corners ordered clockwise
                    starting from any position
        
        Returns:
            tuple: (length, width, orientation) where orientation is in radians
        """
        # Calculate all edge lengths clockwise
        edges = []
        for i in range(4):
            next_idx = (i + 1) % 4
            edge = corners[next_idx] - corners[i]
            edges.append(edge)
            
        # Calculate lengths of all edges
        lengths = [np.linalg.norm(edge) for edge in edges]
        
        # Sort lengths - longer ones are length, shorter ones are width
        sorted_lengths = sorted(lengths, reverse=True)
        length = np.mean(sorted_lengths[:2])  # average of two longest edges
        width = np.mean(sorted_lengths[2:])   # average of two shortest edges
        
        # Find longest edge for orientation
        longest_edge_idx = lengths.index(max(lengths))
        orientation = np.arctan2(edges[longest_edge_idx][1], 
                            edges[longest_edge_idx][0])
        
        return length, width, orientation
    
    def filter_cuboids_by_roi(self, corners: np.ndarray) -> np.ndarray:
        """
        Filter cuboids based on whether they fall within specified ROI.

        Args:
            corners: numpy array of shape (N, 4, 2) containing corner coordinates
            x_range: tuple of (min_x, max_x) defining ROI x bounds (0,70)
            y_range: tuple of (min_y, max_y) defining ROI y bounds  (-40,40)

        Returns:
            numpy array containing only cuboids that fall within ROI
        """
        x_min = self.geometry['W1']
        x_max = self.geometry['W2']
        y_min = self.geometry['L1']
        y_max = self.geometry['L2']
        
        filtered_cuboids = []
        for cuboid in corners:
            # Check if any corner falls within ROI
            if np.any((cuboid[:, 0] >= x_min) & 
                    (cuboid[:, 0] <= x_max) & 
                    (cuboid[:, 1] >= y_min) & 
                    (cuboid[:, 1] <= y_max)):
                filtered_cuboids.append(cuboid)
        
        return np.array(filtered_cuboids)
    
    def _extract_face_corners(self, cuboids: np.ndarray, bottom_face=True):
        """
        Extract corner coordinates of top or bottom face from cuboids.
        
        Args:
            cuboids: numpy array of shape (N, 8, 3) containing cuboid corner coordinates
            bottom_face: bool, if True return bottom face corners, else top face corners
        
        Returns:
            numpy array of shape (N, 4, 2) containing x,y coordinates of face corners
            
                5------4
                |\\    |\\
                | \\   | \\
                6--\\--7  \\
                \\  \\  \\ \\
            l    \\  1-------0    h
            e    \\ ||   \\ ||   e
            n    \\||    \\||   i
            g    \\2------3    g
                t      width.     h
                h.               t
        """
        # Select indices for bottom or top face
        face_index = [0, 1, 5, 4] if bottom_face else [3, 2, 6, 7]
        
        # Extract corners for selected face (x,y coordinates only)
        face_corners = cuboids[:, face_index, :2]
        
        return face_corners
    
    def get_rand_velo(self):
        import random
        rand_v = random.choice(self.velo)
        print("A Velodyne Scan has shape ", rand_v.shape)
        return random.choice(self.velo)

    def load_velo_scan(self, item: int) -> np.ndarray:
        """
        Helper method to get a lidar frame
        
        Args:
            item: The index of the frame to get
        
        Returns:
            A numpy array of shape (N,5)
            Colmuns => x, y, z, laser_number, intensity
        """
        log_id, frame_id = self.global_to_scene_frame[item]
        
        frame_path = self.av2_api.get_lidar_fpath(log_id,frame_id)
        lidar_frame_feather = pd.read_feather(frame_path)
        scan =  lidar_frame_feather[['x', 'y', 'z','laser_number','intensity']].values
        
        return scan

    
    def load_velo(self):
        """Precompute mapping to fill the global_to_scene_frame list"""
        for scene_id in self.scenes:
            frames = self.av2_api.get_ordered_log_lidar_timestamps(scene_id)
            num_frames = len(frames)
            for frame_idx in range(num_frames):
                self.global_to_scene_frame.append((scene_id, frames[frame_idx]))
        self.global_to_scene_frame = self.global_to_scene_frame[::10] # select every 10th sequence to speed up training in av2
        self.total_frames = len(self.global_to_scene_frame)  
        print(f"Total frames: {self.total_frames}")
        print("Done pre-computing the mapping")

    def point_in_roi(self, point):
        if (point[0] - self.geometry['W1']) < 0.01 or (self.geometry['W2'] - point[0]) < 0.01:
            return False
        if (point[1] - self.geometry['L1']) < 0.01 or (self.geometry['L2'] - point[1]) < 0.01:
            return False
        if (point[2] - self.geometry['H1']) < 0.01 or (self.geometry['H2'] - point[2]) < 0.01:
            return False
        return True

    def passthrough(self, velo):
        """
        Helper method to filter out points that are not in the region of interest
        
        Args:
            velo: A numpy array of shape (n, 4) containing the lidar scan
            Columns => x, y, z, intensity
        
        Returns:
            A numpy array of shape (n, 4) containing the lidar scan
        """
        geom = self.geometry
        q = (geom['W1'] < velo[:, 0]) * (velo[:, 0] < geom['W2']) * \
            (geom['L1'] < velo[:, 1]) * (velo[:, 1] < geom['L2']) * \
            (geom['H1'] < velo[:, 2]) * (velo[:, 2] < geom['H2'])
        indices = np.where(q)[0]
        return velo[indices, :]

    def lidar_preprocess(self, scan: np.ndarray) -> np.ndarray:
        """
        Voxelizes the lidar scan
        
        Args:
            scan: A numpy array of shape (n, 4) containing the lidar scan
            Columns => x, y, z, intensity
        
        Returns:
            A numpy array of shape (36, 800, 700) containing the voxelized lidar scan
        """
        velo_processed = np.zeros(self.geometry['input_shape'], dtype=np.float32)
        intensity_map_count = np.zeros((velo_processed.shape[0], velo_processed.shape[1]))
        velo = self.passthrough(scan)
        for i in range(velo.shape[0]):
            x = int((velo[i, 1]-self.geometry['L1']) / 0.1)
            y = int((velo[i, 0]-self.geometry['W1']) / 0.1)
            z = int((velo[i, 2]-self.geometry['H1']) / 0.1)
            velo_processed[x, y, z] = 1
            velo_processed[x, y, -1] += velo[i, 3]
            intensity_map_count[x, y] += 1
        velo_processed[:, :, -1] = np.divide(velo_processed[:, :, -1],  intensity_map_count,
                                             where=intensity_map_count != 0)
        return velo_processed
    

def get_data_loader(batch_size, use_npy, geometry=None):
    train_dataset = AV2(train=True)
    if geometry is not None:
        train_dataset.geometry = geometry
    train_dataset.load_velo()
    train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    
    val_dataset = AV2(train=False)
    if geometry is not None:
        val_dataset.geometry = geometry
    val_dataset.load_velo()
    val_data_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size * 4, num_workers=0)
    print(f"Total frames in train dataset: {len(train_dataset)}")
    print(f"Total frames in validation dataset: {len(val_dataset)}")
    print("------------------------------------------------------------------")
    return train_data_loader, val_data_loader

def test():
    # npy average time 0.31s
    # c++ average time 0.08s 4 workers
    batch_size = 3
    train_data_loader, val_data_loader = get_data_loader(batch_size, False)
    times = []
    tic = time.time()
    for i, (input, label_map, item) in enumerate(train_data_loader):
        toc = time.time()
        print(toc - tic)
        times.append(toc-tic)
        tic = time.time()
        print("Entry", i)
        print("Input shape:", input.shape)
        print("Label Map shape", label_map.shape)
        if i == 10:
            break
    print("average preprocess time per image", np.mean(times)/batch_size)    

    print("Finish testing train dataloader")

if __name__ == "__main__":
    test()