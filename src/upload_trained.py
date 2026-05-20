from common_tools.s3_operations import *


def main():
    dataset_idx = 1
    upload_model_to_s3(
        f"model_weights/weights_{dataset_idx}.pth",
        "models",
        f"resnet18/weights_{dataset_idx}.pth",
    )

    return 0


if __name__ == "__main__":
    main()
