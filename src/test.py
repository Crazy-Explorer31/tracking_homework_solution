import json

from common_tools.data_operations import *
from common_tools.model_operations import *
import yaml
from common_tools.s3_operations import *
from time import time


def main():
    params = load_params()
    learning_rate, batch_size, num_epochs, path_to_result_weights, dataset_idx = (
        float(params["learning_rate"]),
        int(params["batch_size"]),
        int(params["num_epochs"]),
        params.get("path_to_result_weights"),
        int(params["dataset_idx"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = get_test_dataloader(batch_size=batch_size, dataset_idx=dataset_idx)

    model = get_model_local(device, path_to_result_weights)

    criterion = nn.CrossEntropyLoss()

    model.eval()

    test_loss, test_acc, test_f1, test_precision, test_recall = validate(
        model, test_loader, criterion, device
    )

    hparams = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "dataset_idx": dataset_idx,
    }
    metrics = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "test_precision": test_precision,
        "test_recall": test_recall,
    }

    print(metrics)

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    writer = SummaryWriter(f"my_logs/resnet18_on_dataset_{dataset_idx}")
    writer.add_hparams(hparam_dict=hparams, metric_dict=metrics)

    return 0


if __name__ == "__main__":
    main()
