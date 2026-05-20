from common_tools.imports import *
import yaml


def load_params(path="params.yaml"):
    with open(path, "r") as f:
        params = yaml.safe_load(f)
    return params


class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.root_dir / self.data.iloc[idx]["file_name"]
        image = Image.open(img_path).convert("RGB")
        label = self.data.iloc[idx]["label"]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_train_dataloader(batch_size=32, dataset_idx=1):
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = ImageDataset(
        csv_file=f"ai-vs-human-generated-dataset-hw/Train_{dataset_idx}/train.csv",
        root_dir=f"ai-vs-human-generated-dataset-hw/Train_{dataset_idx}",
        transform=train_transform,
    )

    train_dataset = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    print(f"Train dataset size: {len(train_dataset)}")

    return train_dataset


def get_test_dataloader(batch_size=32, dataset_idx=1):
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_dataset = ImageDataset(
        csv_file=f"ai-vs-human-generated-dataset-hw/Test_{dataset_idx}/test.csv",
        root_dir=f"ai-vs-human-generated-dataset-hw/Test_{dataset_idx}",
        transform=test_transform,
    )

    test_dataset = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    print(f"Test dataset size: {len(test_dataset)}")

    return test_dataset
