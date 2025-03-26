import torch
import json
import pickle
import numpy as np
from torch.utils.data import Dataset

class SurfActionDataset(Dataset):
    def __init__(self, data_root, split_list, clip_len=243):
        """
        data_root: folder containing .pkl and .json files
        split_list: list of video names to include (train or val split)
        """
        self.data_root = data_root
        self.video_paths = split_list
        self.clip_len = clip_len
        self.samples = self.load_all_data()

    def load_all_data(self):
        all_data = []
        for video_path in self.video_paths:
            json_path = f"{video_path}/keypoints_xyz.json"
            pkl_path = f"{video_path}/input_2D/keypoints.npz"
            # Load 3D keypoints from JSON
            with open(json_path, 'r') as f:
                json_data = json.load(f)

            frames = json_data["frames"]
            label_name = json_data["label"]
            label_id = self.label_to_index(label_name)

            # Load 2D keypoints from PKL
            with open(pkl_path, 'rb') as f:
                kp2d = pickle.load(f)  # Shape: (T, 17, 2)

            T = min(len(frames), len(kp2d))
            if T < self.clip_len:
                continue  # Skip short clips

            # Sample uniformly
            for start in range(0, T - self.clip_len + 1, self.clip_len):
                pose2d_clip = []
                pose3d_clip = []

                for i in range(start, start + self.clip_len):
                    # 2D
                    kp_2d = kp2d[i]  # (17, 2)
                    kp_2d = np.pad(kp_2d, ((0,0),(0,1)))  # to (17,3)

                    # 3D
                    frame = frames[i]["keypoints"]
                    kp_3d = np.array([[frame[str(j)]["x"],
                                       frame[str(j)]["y"],
                                       frame[str(j)]["z"]] for j in range(17)])  # (17,3)

                    pose2d_clip.append(kp_2d)
                    pose3d_clip.append(kp_3d)

                # Stack to (T, 17, 3)
                pose2d_clip = np.stack(pose2d_clip)
                pose3d_clip = np.stack(pose3d_clip)

                # Stack into (2, T, 17, 3)
                combined = np.stack([pose2d_clip, pose3d_clip], axis=0)

                all_data.append((combined, label_id))

        return all_data

    def label_to_index(self, label):
        label_map = {
            "360": 0,
            "cutback-frontside": 1,
            "roller": 1,
            "take-off": 2
        }
        return label_map[label]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_tensor, label = self.samples[idx]
        return torch.tensor(input_tensor, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
