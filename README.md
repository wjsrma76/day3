# 1.Transfor Learning 을 하기 위한 코드와 설명
 학습된 모델 일부분(lower layer)을 재사용하여 모델 B를 학습 시킬 수 있다. 이러한 방법을 Transfer Learning이라고 한다


for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False


# 2.Fine Tuning을 하기 위한 코드와 설명

history=model.fit(train_generator,  
                    epochs=5, 
validation_data=valid_generator)



# 3. 에포크 시마다, 가장 좋은 모델을 저장하는,ModelCheckpoint

from tensorflow.keras.callbacks import ModelCheckpoint
cp = ModelCheckpoint(filepath=CHECKPOINT_PATH,monitor='val_accuracy',save_best_only=True,verbose= 1)


# 4.에포크 시마다, 기록을 남길수 있는,CSVLogger 사용 방법
from tensorflow.keras.callbacks import CSVLogger
csv_logger = CSVLogger(filename=LOGFILE_PATH,append=True)
