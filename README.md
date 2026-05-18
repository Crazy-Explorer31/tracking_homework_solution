## Решение ДЗ MLOPS

1. Решения в `train_model.ipynb`
2. Решения в `train_model.ipynb`
3. Решения в `train_model.ipynb`

4. Решение состоит из следующих файлов:
* `src/*.py`
* `dvc.yaml`
* `params.yaml`


Для воспроизведения решения пункта 4 необходимо:
* Запустить tensorboard командой `tensorboard --logdir=./my_logs`
* Запустить s3-хранилище командой `docker-compose up --build`
* * Загрузить в бакет `models` по пути `resnet18/weights_0.pth` веса модели resnet18
* В корне проекта выполнить `dvc init`
* Выполнить первую стадию обучения:
* * Переключиться на исходники первой стадии `git checkout m1.0`
* * Запустить пайплайн обучения `dvc repro`
* Выполнить вторую стадию обучения:
* * Переключиться на исходники второй стадии `git checkout m2.0`
* * Запустить пайплайн обучения `dvc repro`
* Наблюдать результаты в tensorboard на http://localhost:6006
