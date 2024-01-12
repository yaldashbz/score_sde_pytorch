import os
import json
import torch
import argparse
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from torch.utils.data import Subset


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2. - 1.
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.) / 2.
    else:
        return lambda x: x


class MidogDataset(Dataset):
    def __init__(self, root_dir, json_file, transform=None, scanner=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = os.listdir(root_dir)
        self.scanner = scanner

        with open(json_file, 'r') as f:
            self.coco_data = json.load(f)

        if self.scanner:
            self.image_files = [
                im['file_name'].split('/')[-1] for im in self.coco_data['images']
                if im['scanner'] == scanner
            ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])

        # image 1036 is corrupted, use 1035 instead
        if '1036' in img_name and 'train' in self.root_dir:
            img_name = img_name.replace('1036', '1035')

        image = Image.open(img_name)
        label = [im['scanner'] for im in self.coco_data['images']
                 if im['file_name'].split('/')[-1] == img_name.split('/')[-1]][0]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(int(label) - 1)


train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
])


def cli():
    parser = argparse.ArgumentParser(description='Score-Based SDE')
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--test-batch-size', default=4, type=int)
    parser.add_argument('--root', default='/content/drive/My Drive/MIDOG/dataset_v7', type=str, metavar='PATH')
    return parser.parse_args()

def main():

    args = cli()
    root = args.root
    train_dir = os.path.join(root, 'train')
    eval_dir = os.path.join(root, 'eval')
    test_dir = os.path.join(root, 'test')

    batch_size = args.batch_size
    test_batch_size = args.test_batch_size

    midog_train_ds = MidogDataset(
        root_dir=train_dir, json_file=os.path.join(root, 'train.json'),
        transform=train_transforms
    )
    train_data_loader = DataLoader(midog_train_ds, batch_size=batch_size, shuffle=True)

    midog_val_ds = MidogDataset(
        root_dir=eval_dir, json_file=os.path.join(root, 'eval.json'),
        transform=test_transforms
    )
    val_data_loader = DataLoader(midog_val_ds, batch_size=batch_size, shuffle=False)

    test_loaders = []
    scanners = [1, 2, 3, 4]
    for scanner in scanners:
        midog_test_ds = MidogDataset(
            root_dir=test_dir, json_file=os.path.join(root, 'test.json'),
            transform=test_transforms, scanner=scanner
        )
        test_data_loader = DataLoader(
            midog_test_ds, batch_size=test_batch_size, shuffle=True)  # for random sample
        test_loaders.append(test_data_loader)

    test_dataset = ConcatDataset([Subset(test_loader.dataset, indices=range(100)) for test_loader in test_loaders])
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_data_loader, val_data_loader, test_data_loader, test_loaders