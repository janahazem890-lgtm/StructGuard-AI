import os
import glob
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T

class CrackDataset(Dataset):
    def __init__(self, file_pairs, img_size=(256, 256), is_train=False):
        self.file_pairs = file_pairs
        self.img_size = img_size
        self.is_train = is_train

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.file_pairs[idx]
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # Resize both to fixed size
        image = TF.resize(image, self.img_size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.img_size, interpolation=TF.InterpolationMode.NEAREST)

        # Augmentations for train set
        if self.is_train:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)
            # Brightness jitter on image only
            if random.random() > 0.5:
                factor = random.uniform(0.8, 1.2)
                image = TF.adjust_brightness(image, factor)

        # Convert image to tensor & normalize
        img_t = TF.to_tensor(image) # [0, 1]
        img_t = TF.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Convert mask to binary tensor (0.0 or 1.0)
        mask_t = TF.to_tensor(mask) # [1, H, W], values in [0, 1]
        mask_t = (mask_t > 0.5).float()

        return img_t, mask_t, img_path

def get_all_pairs():
    # Gather pairs from train_img/train_lab and test_img/test_lab
    pairs = []
    for split in ['train', 'test']:
        img_dir = os.path.join('data', 'raw', f'{split}_img')
        lab_dir = os.path.join('data', 'raw', f'{split}_lab')
        if not os.path.exists(img_dir):
            continue
        for f in os.listdir(img_dir):
            base, ext = os.path.splitext(f)
            mask_name = base + '.png'
            mask_path = os.path.join(lab_dir, mask_name)
            img_path = os.path.join(img_dir, f)
            if os.path.exists(mask_path):
                pairs.append((img_path, mask_path))
    return sorted(pairs)

def get_data_splits(split_json_path='data/splits.json', seed=42):
    if os.path.exists(split_json_path):
        with open(split_json_path, 'r') as f:
            splits = json.load(f)
            return splits['train'], splits['val'], splits['test']

    all_pairs = get_all_pairs()
    random.seed(seed)
    random.shuffle(all_pairs)

    n_total = len(all_pairs)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    
    train_pairs = all_pairs[:n_train]
    val_pairs = all_pairs[n_train:n_train + n_val]
    test_pairs = all_pairs[n_train + n_val:]

    splits = {
        'train': train_pairs,
        'val': val_pairs,
        'test': test_pairs
    }
    os.makedirs(os.path.dirname(split_json_path), exist_ok=True)
    with open(split_json_path, 'w') as f:
        json.dump(splits, f, indent=2)

    return train_pairs, val_pairs, test_pairs

if __name__ == '__main__':
    train_p, val_p, test_p = get_data_splits()
    print(f'Total: {len(train_p) + len(val_p) + len(test_p)} pairs')
    print(f'Train: {len(train_p)}, Val: {len(val_p)}, Test: {len(test_p)}')
    ds = CrackDataset(train_p, is_train=True)
    img, mask, path = ds[0]
    print(f'Dataset sample 0: img shape={img.shape}, mask shape={mask.shape}, mask unique={torch.unique(mask)}')
