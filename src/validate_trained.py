import json

from common_tools.data_operations import *
from common_tools.s3_operations import *
from common_tools.model_operations import ResNetModelWrapper, get_model_local, validate


def main():
    params = load_params()
    batch_size = int(params["train"]["batch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    dataset_idx = 1

    test_loader = get_test_dataloader(batch_size=batch_size, dataset_idx=dataset_idx)

    model = get_model_local(device, f"model_weights/weights_{dataset_idx}.pth")
    model.eval()

    test_loss, test_acc, test_f1, test_precision, test_recall = validate(
        model, test_loader, criterion, device
    )

    metrics = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "test_precision": test_precision,
        "test_recall": test_recall,
    }

    print(metrics)

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/metrics_of_trained.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # TODO: log to tensorboard

    return 0


if __name__ == "__main__":
    main()
