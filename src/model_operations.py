from common import *
from s3_operations import *


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

    print(f"Model architecture:")
    print(model)

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
