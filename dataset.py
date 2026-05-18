import os
import tarfile
import requests
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np
# 下载
def download_with_progress(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"{os.path.basename(path)} already exists, skip download.")
        return

    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))

    with open(path, "wb") as file, tqdm(
        desc=os.path.basename(path),
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))

# 解压
def extract_tar(path, extract_to):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    with tarfile.open(path) as tar:
        first_member_name = tar.getmembers()[0].name
        first_dir = first_member_name.split('/')[0]

        if os.path.exists(os.path.join(extract_to, first_dir)):
            print(f"{first_dir} already extracted, skip.")
            return

        print(f"Extracting {path} ...")
        tar.extractall(path=extract_to)


def download_and_extract(root):
    os.makedirs(root, exist_ok=True)

    images_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    masks_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

    images_tar = os.path.join(root, "images.tar.gz")
    masks_tar = os.path.join(root, "annotations.tar.gz")

    download_with_progress(images_url, images_tar)
    download_with_progress(masks_url, masks_tar)

    extract_tar(images_tar, root)
    extract_tar(masks_tar, root)

    print("Dataset ready!")

#  Dataset
class OxfordPetDataset(Dataset):
    def __init__(self, root="data", split="train", img_size=256, val_split=0.1, seed=42):
        download_and_extract(root)

        self.root = root
        self.img_size = img_size

        image_dir = os.path.join(root, "images")
        mask_dir = os.path.join(root, "annotations", "trimaps")

        trainval_file = os.path.join(root, "annotations", "trainval.txt")
        test_file = os.path.join(root, "annotations", "test.txt")

        with open(trainval_file, "r") as f:
            trainval_ids = [line.strip().split()[0] for line in f]

        train_ids, val_ids = train_test_split(trainval_ids, test_size=val_split, random_state=seed)


        # train_ids = train_ids[:200]
        # val_ids = val_ids[:50]


        with open(test_file, "r") as f:
            test_ids = [line.strip().split()[0] for line in f]

        if split == "train":
            ids = train_ids
        elif split == "val":
            ids = val_ids
        elif split == "test":
            ids = test_ids
        else:
            raise ValueError("split must be train / val / test")

        self.images = [i + ".jpg" for i in ids]
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # self.mask_transform = transforms.Compose([
        #     transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
        #     transforms.PILToTensor()
        # ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", ".png"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # image
        image = self.img_transform(image)

        # mask
        mask = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)
        mask = torch.from_numpy(np.array(mask)).long()

        # 类别修正
        mask = mask - 1
        mask = torch.clamp(mask, 0, 2)

        return image, mask

# -------------------------------
# 5. 获取 DataLoader
# -------------------------------
# def get_train_val_dataloaders(root='data', batch_size=8, image_size=256, num_workers=4, val_split=0.1):
#     full_dataset = OxfordPetDataset(root=root, split='train', img_size=image_size, val_split=val_split)
#     indices = list(range(len(full_dataset)))
#     train_indices, val_indices = train_test_split(indices, test_size=val_split, random_state=42)

#     train_loader = DataLoader(
#         Subset(full_dataset, train_indices),
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=True
#     )

#     val_loader = DataLoader(
#         Subset(full_dataset, val_indices),
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True
#     )

#     return train_loader, val_loader
def get_train_val_dataloaders(root='data', batch_size=8, image_size=256, num_workers=4, val_split=0.1):
    # 1. 严格获取内部划分好的纯训练集
    train_dataset = OxfordPetDataset(root=root, split='train', img_size=image_size, val_split=val_split)
    # 2. 严格获取内部划分好的纯验证集
    val_dataset = OxfordPetDataset(root=root, split='val', img_size=image_size, val_split=val_split)

    train_loader = DataLoader(
        train_dataset,  # 直接用 train_dataset，去掉 Subset
        batch_size=batch_size,
        shuffle=True,   # 训练集需要打乱
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,    # 直接用 val_dataset，去掉 Subset
        batch_size=batch_size,
        shuffle=False,  # 验证集不需要打乱
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader

def get_test_dataloader(root='data', batch_size=8, image_size=256, num_workers=4):
    test_dataset = OxfordPetDataset(root=root, split='test', img_size=image_size)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return test_loader

# -------------------------------
# 6. 测试代码
# -------------------------------
if __name__ == "__main__":
    train_loader, val_loader = get_train_val_dataloaders()
    test_loader = get_test_dataloader()

    for imgs, masks in train_loader:
        print("Batch images:", imgs.shape)
        print("Batch masks:", masks.shape)
        break