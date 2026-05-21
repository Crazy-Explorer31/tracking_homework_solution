## Решение ДЗ MLOPS

1. Решения в `train_model.ipynb`
2. Решения в `train_model.ipynb`
3. Решения в `train_model.ipynb`

4. Решение состоит из следующих файлов:
* `src/*.py`
* `dvc.yaml`
* `params.yaml`


Для воспроизведения решения пункта 4 необходимо:
* Скачать [датасет](https://s138klg.storage.yandex.net/rdisk/df0f4f5de3131bc65a33319caf272f2a497cb8c708f75e7f03c53b967546afbc/6a075700/Ae-VyIrOZFIdIgL48IxyZnFOTddc1vSAbnOEzGLwttUkJoEVc1hEh-foL8KYEuUDlQclvUu_GKBg1RaSJw1ivw==?uid=0&filename=ai-vs-human-generated-dataset-hw.zip&disposition=attachment&hash=83AxKETz95dpM5ocUNL4gKxU0HoeKAqV/C/XcRStAxNNXfty8qLffD4yzApoevrAq/J6bpmRyOJonT3VoXnDag%3D%3D&limit=0&content_type=application%2Fzip&owner_uid=1062841563&fsize=1149185420&hid=06f715004a5f4d6329b202750c047b9c&media_type=compressed&tknv=v3&ts=651de7f43c000&s=19435edd9bfd035f05b4da09a7845c73d8b3da1c8ae79a06aac722cb49e82e1d&pb=U2FsdGVkX19i9A5MEA7OWuq1TXo13RmxoRVYpgzmRuGKLrnEjl88Rw4W_PCTSPHEGGu7IB0FVZojElVMZvl1usOAQ59I8wWYODYIcRzNrFA)
* Сделать его `unzip` в директорию проекта
* Переименовать поддиректории датасета `test_data` в `train_data`
* Положить по пути `model_weights/weights_0.pth` [веса](https://download.pytorch.org/models/resnet18-f37072fd.pth) модели resnet18

* Запустить tensorboard командой `tensorboard --logdir=./my_logs`
* Запустить s3-хранилище командой `docker-compose up --build`
* При желании добавить s3-remote для dvc:
* * `dvc remote add -d s3_store s3://dvc/dvc-store`
* * `dvc remote modify s3_store endpointurl http://localhost:9000`
* * `dvc remote modify s3_store addressing_style path`
* * `dvc remote modify --local s3_store access_key_id 'minioadmin'`
* * `dvc remote modify --local s3_store secret_access_key 'minioadmin'`

* В корне проекта выполнить `dvc init`
* Запустить пайплайн обучения `dvc repro`
* Наблюдать результаты в tensorboard на http://localhost:6006

### Граф зависимостей DVC
```
                                 +-------+                                                    
                               **| train |****                                                
                          *****  +-------+    ********                                        
                      ****          *                 ********                                
                 *****             *                          ********                        
              ***                  *                                  ********                
+----------------+         +------------+                                     *****           
| upload_trained |         | post_train |                                         *           
+----------------+         +------------+*                                        *           
                         **               **                                      *           
                      ***                   ***                                   *           
                    **                         **                                 *           
      +---------------------+         +-----------------------+         +------------------+  
      | upload_post_trained |         | validate_post_trained |         | validate_trained |  
      +---------------------+         +-----------------------+         +------------------+  
                                                          **             ***                  
                                                            ***        **                     
                                                               **    **                       
                                                         +----------------+                   
                                                         | compare_models |                   
                                                         +----------------+                   
```
