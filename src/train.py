from common_tools.data_operations import *
from common_tools.s3_operations import *
from common_tools.model_operations import ResNetModelWrapper


def main():
    params = load_params()
    learning_rate, batch_size, num_epochs = (
        float(params["post_train"]["learning_rate"]),
        int(params["post_train"]["batch_size"]),
        int(params["post_train"]["num_epochs"]),
    )
    dataset_idx = 1

    model_wrapper = ResNetModelWrapper(
        f"model_weights/weights_{dataset_idx - 1}.pth",
        learning_rate,
        batch_size,
        num_epochs,
    )
    model_wrapper.train(dataset_idx)
    model_wrapper.dump_model(dataset_idx)

    return 0


if __name__ == "__main__":
    main()
