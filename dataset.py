import os
import glob
import numpy as np
from numpy.random import randint
from torch.utils import data
import torch
import torchvision.transforms as transforms
from PIL import Image

class VideoDataset(data.Dataset):
    def __init__(self, records, num_segments=16, duration=1, mode='train', image_size=112):
        self.records = records
        self.num_segments = num_segments
        self.duration = duration
        self.mode = mode
        self.image_size = image_size
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def _get_train_indices(self, num_frames):
        average_duration = (num_frames - self.duration + 1) // self.num_segments
        if average_duration > 0:
            return np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif num_frames > self.num_segments:
            return np.sort(randint(num_frames - self.duration + 1, size=self.num_segments))
        return np.zeros((self.num_segments,))

    def _get_test_indices(self, num_frames):
        if num_frames > self.num_segments + self.duration - 1:
            tick = (num_frames - self.duration + 1) / float(self.num_segments)
            return np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        return np.zeros((self.num_segments,))

    def __getitem__(self, index):
        path, num_frames, label, video_name, subj = self.records[index]
        extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
        video_frames_path = []
        for ext in extensions:
            video_frames_path.extend(glob.glob(os.path.join(path, ext)))
        video_frames_path.sort()
        
        indices = self._get_train_indices(num_frames) if self.mode == 'train' else self._get_test_indices(num_frames)
        
        images = []
        for seg_ind in indices:
            p = int(seg_ind)
            if p >= len(video_frames_path):
                p = len(video_frames_path) - 1
            img = Image.open(video_frames_path[p]).convert('RGB')
            if self.mode == 'train' and randint(0, 2) == 1:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            images.append(self.transform(img))
            
        tensor_images = torch.stack(images)
        return tensor_images, torch.tensor(label, dtype=torch.long), video_name

    def __len__(self):
        return len(self.records)

def build_mmew_records(data_root="/home/ab3379/work/data/MMEW_Final/Macro_Expression"):
    classes = sorted(['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise'])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    video_records = []
    
    for subj in os.listdir(data_root):
        subj_path = os.path.join(data_root, subj)
        if not os.path.isdir(subj_path):
            continue
        for emo in os.listdir(subj_path):
            emo_path = os.path.join(subj_path, emo)
            if not os.path.isdir(emo_path) or emo not in class_to_idx:
                continue
            frames = glob.glob(os.path.join(emo_path, "*.jpg")) + glob.glob(os.path.join(emo_path, "*.png"))
            if len(frames) >= 5:
                # Store (path, num_frames, label, video_name, subject_id)
                video_records.append((emo_path, len(frames), class_to_idx[emo], f"{subj}_{emo}", subj))
                
    print(f"Loaded {len(video_records)} valid MMEW Macro-Expression sequences across subjects.")
    return video_records, classes