from common_tools.data_operations import get_train_dataloader
from common_tools.imports import *
from common_tools.s3_operations import *

# import spdlog


def get_model(device, path_to_weigths=None):
    are_pure_weights = (
        path_to_weigths is not None and path_to_weigths.find("weights_0.pth") != -1
    )
    are_used_weights = (
        path_to_weigths is not None and path_to_weigths.find("weights_0.pth") == -1
    )

    model = models.resnet18()
    if are_pure_weights:
        loaded_state_dict = load_model_state_from_s3("models", path_to_weigths)
        model.load_state_dict(loaded_state_dict)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    if are_used_weights:
        loaded_state_dict = load_model_state_from_s3("models", path_to_weigths)
        model.load_state_dict(loaded_state_dict)
    model = model.to(device)

    print(f"Model constructed")

    return model


def get_model_local(device, path_to_weigths):
    are_pure_weights = (
        path_to_weigths is not None and path_to_weigths.find("weights_0.pth") != -1
    )
    are_used_weights = (
        path_to_weigths is not None and path_to_weigths.find("weights_0.pth") == -1
    )

    model = models.resnet18()
    if are_pure_weights:
        model.load_state_dict(torch.load(path_to_weigths))
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    if are_used_weights:
        model.load_state_dict(torch.load(path_to_weigths))
    model = model.to(device)

    print(f"Model constructed")

    return model


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # подсчет epoch_loss, epoch_acc, epoch_f1
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds)

    return epoch_loss, epoch_acc, epoch_f1


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # подсчет epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds)
    epoch_precision = precision_score(all_labels, all_preds)
    epoch_recall = recall_score(all_labels, all_preds)

    return epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall


class ResNetModelWrapper:

    def __init__(
        self,
        path_to_weights,
        learning_rate,
        batch_size,
        num_epochs,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = get_model_local(self.device, path_to_weights)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.1
        )

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def train(self, dataset_idx):
        train_losses = []
        train_accs = []
        train_f1s = []

        train_loader = get_train_dataloader(
            batch_size=self.batch_size, dataset_idx=dataset_idx
        )

        writer = SummaryWriter(f"my_logs/resnet18_stage_{dataset_idx}")

        self.model.train()
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-" * 50)

            train_loss, train_acc, train_f1 = train_epoch(
                self.model, train_loader, self.criterion, self.optimizer, self.device
            )
            self.scheduler.step()

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            train_f1s.append(train_f1)

            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("acc/train", train_acc, epoch)
            writer.add_scalar("f1/train", train_f1, epoch)

            print(
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}"
            )

        print("\nTraining completed!")

    def dump_model(self, dataset_idx):
        os.makedirs("model_weights", exist_ok=True)
        torch.save(self.model.state_dict(), f"model_weights/weights_{dataset_idx}.pth")
