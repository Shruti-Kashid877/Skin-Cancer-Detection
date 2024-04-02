import numpy as np
import pandas as pd
import os
import glob
import tensorflow as tf
from tensorflow import keras
import random
from sklearn.metrics import accuracy_score

cancer=pd.read_csv("D:/7Cancer/HAM10000_metadata.csv")

cancer.fillna({'age': np.mean(cancer['age'])}, inplace=True)
image_paths=[]
for part in ("part_1","part_2"):
    image_paths+=glob.glob("D:/7Cancer/HAM10000_images_"+part+"/*")
image_ids_n_paths = {os.path.splitext(os.path.basename(path))[0]:path for path in image_paths}
cancer['path']=cancer['image_id'].map(image_ids_n_paths)
labels=cancer['dx'].to_frame()
cancer=cancer.drop('dx',axis=1)
labels=pd.get_dummies(labels)
cancer.head()
labels.head()
preliminary_data=list(zip(cancer['path'],labels.values.tolist()))
random.shuffle(preliminary_data)
paths,labels=zip(*preliminary_data)
data=tf.data.Dataset.from_tensor_slices((list(paths),list(labels)))
def final_data(path,label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.image.resize(image,[90,120])
    image = image/255
    return image,label

data=data.map(final_data).prefetch(30)


train_size=round(0.8*10015)
val_size=round(0.1*10015)
test_size=10015-train_size-val_size

train=data.take(train_size)
val=data.skip(train_size)
test=data.skip(train_size)
val=data.take(val_size)
test=data.take(test_size)

mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/cpu:0"])

with mirrored_strategy.scope():
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(30,(5,5),strides=(1,1),padding='valid',activation='relu',input_shape=(90,120,3)))
    model.add(tf.keras.layers.Conv2D(30,(3,3),strides=(1,1),padding='valid',activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=None,padding='valid'))
    model.add(tf.keras.layers.Conv2D(20,(3,3),strides=(1,1),padding='valid',activation='relu'))
    model.add(tf.keras.layers.Conv2D(15,(3,3),strides=(1,1),padding='valid',activation='relu'))
    model.add(tf.keras.layers.Conv2D(15,(3,3),strides=(1,1),padding='valid',activation='relu'))
    model.add(tf.keras.layers.GroupNormalization(groups=3))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=None,padding='valid'))
    model.add(tf.keras.layers.Conv2D(10,(3,3),strides=(1,1),padding='valid',activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Normalization())
    model.add(tf.keras.layers.Dense(256,activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(128,activation='relu'))
    model.add(tf.keras.layers.Dense(7,activation='softmax'))
    model.compile(optimizer="Adam",loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

with mirrored_strategy.scope():
    checkpoint= tf.keras.callbacks.ModelCheckpoint(
        filepath='D:/7Cancer/skin_cancer_detection71.h5',
        save_weights_only=False,
        monitor='val_accuracy',
        save_best_only=True,
        save_freq="epoch",
        )
    early_stopping= tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
    def lr_scheduler(epoch,lr,epochs=50):
        initial=1e-3
        if epoch<epochs*0.1:
            return initial
        elif epoch>epochs*0.1 and epoch<epochs*0.25:
            lr*=tf.math.exp(-0.1)
            return lr
        else:
            lr*=tf.math.exp(-0.008)
            return lr
    lr_scheduling=tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

with mirrored_strategy.scope():
    history = model.fit(train.batch(60),epochs=50,validation_data=val.batch(60),
                        callbacks=[checkpoint,early_stopping,lr_scheduling],shuffle=True)



model1=tf.keras.models.load_model("D:/7Cancer/skin_cancer_detection7.h5")

predictions=model1.predict(test.batch(len(test)))

predictions[0]

def outputs(x):
    a = np.zeros(x.shape)
    a[np.where(x==np.max(x))] = 1
    return a

for i in range(len(predictions)):
    predictions[i]=outputs(predictions[i])

predictions

predictions[0]



y_test = np.concatenate([y for x, y in test.batch(len(test))], axis=0)

y_test

accuracy_score=accuracy_score(predictions,y_test)
print("Test accuracy:", accuracy_score)
