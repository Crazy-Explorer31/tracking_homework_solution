from data_operations import *
from model_operations import *
import yaml
from s3_operations import *
from time import time


def main():
    params = load_params()
    (
        learning_rate,
        batch_size,
        num_epochs,
        path_to_initial_weights,
        path_to_result_weights,
        dataset_idx,
    ) = (
        float(params["learning_rate"]),
        int(params["batch_size"]),
        int(params["num_epochs"]),
        params.get("path_to_initial_weights"),  # can be None
        params.get("path_to_result_weights"),
        int(params["dataset_idx"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = get_train_dataloader(batch_size=batch_size, dataset_idx=dataset_idx)

    model = get_model(device, path_to_initial_weights)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # lr
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    num_epochs = num_epochs
    train_losses = []
    train_accs = []
    train_f1s = []

    # Логгирование метрик для tensorboard
    writer = SummaryWriter(f"my_logs/resnet18_on_dataset_{dataset_idx}")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)

        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        scheduler.step()

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_f1s.append(train_f1)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("f1/train", train_f1, epoch)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")

    print("\nTraining completed!")

    # model dump to s3
    os.makedirs("model_weights", exist_ok=True)
    torch.save(model.state_dict(), "model_weights/resnet18.pth")
    upload_model_to_s3("model_weights/resnet18.pth", "models", path_to_result_weights)

    return 0


if __name__ == "__main__":
    main()
