import torch
import io
from pathlib import Path
import os
import boto3
from botocore.config import Config

S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://127.0.0.1:9000")
MINIO_ROOT_USER = os.getenv("MINIO_ROOT_USER", "minioadmin")
MINIO_ROOT_PASSWORD = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
S3_REGION = os.getenv("S3_REGION", "us-east-1")


def get_s3_client(external: bool = False):
    """Возвращает настроенный клиент S3 для MinIO."""
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=MINIO_ROOT_USER,
        aws_secret_access_key=MINIO_ROOT_PASSWORD,
        region_name=S3_REGION,
        config=Config(s3={"addressing_style": "path"}),
    )


def upload_model_to_s3(
    local_path: str, bucket: str, key: str, external: bool = False
) -> None:
    """
    Загружает локальный файл .pth в S3-совместимое хранилище.

    Args:
        local_path: путь к файлу на диске
        bucket: имя бакета
        key: ключ (путь) объекта в бакете
        external: флаг для get_s3_client (не используется в текущей реализации)
    """
    client = get_s3_client(external=external)

    # Убедимся, что бакет существует (если нет – создадим)
    try:
        client.head_bucket(Bucket=bucket)
    except client.exceptions.NoSuchBucket:
        client.create_bucket(Bucket=bucket)
        print(f"Бакет '{bucket}' создан")

    with open(local_path, "rb") as f:
        client.upload_fileobj(f, bucket, key)
    print(f"Модель загружена: s3://{bucket}/{key}")


def download_model_from_s3(
    bucket: str, key: str, local_path: str, external: bool = False
) -> None:
    """
    Скачивает .pth-файл из S3 в локальную файловую систему.

    Args:
        bucket: имя бакета
        key: ключ объекта
        local_path: куда сохранить файл
        external: флаг для get_s3_client
    """
    client = get_s3_client(external=external)

    # Создаём директории, если их нет
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    with open(local_path, "wb") as f:
        client.download_fileobj(bucket, key, f)
    print(f"Model downloaded: {local_path}")


def load_model_state_from_s3(
    bucket: str, key: str, model=None, map_location=None, external: bool = False
):
    """
    Загружает модель (state_dict или полный объект) напрямую из S3 в память.

    Args:
        bucket: имя бакета
        key: ключ объекта
        model: экземпляр модели PyTorch (если None, вернёт загруженный объект)
        map_location: устройство для загрузки (например, torch.device('cuda'))
        external: флаг для get_s3_client

    Returns:
        Если model не None: загружает state_dict и возвращает model
        Если model is None: возвращает то, что сохранено (state_dict или полную модель)
    """
    client = get_s3_client(external=external)

    # Скачиваем в байтовый буфер
    buffer = io.BytesIO()
    client.download_fileobj(bucket, key, buffer)
    buffer.seek(0)

    # Загружаем из буфера
    checkpoint = torch.load(buffer, map_location=map_location)

    if model is not None:
        # Если передан объект модели – загружаем state_dict (поддерживает чекпоинты с 'model_state_dict')
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model
    else:
        # Возвращаем как есть (state_dict или полную модель)
        return checkpoint
