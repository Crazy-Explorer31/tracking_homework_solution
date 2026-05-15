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
* В корне проекта выполнить `dvc init`
* Запустить пайплайн `dvc repro`
* Изменить номер датасета и начальные веса модели в `params.yaml` (следуя инструкциям из файла)
* Запустить пайплайн `dvc repro` повторно
* Наблюдать результаты в tensorboard на http://localhost:6006
