import tensorflow as tf
from tensorflow.keras import layers, Sequential
import matplotlib.pyplot as plt

import argparse
parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parse.add_argument('--train',type=str, default='images/train',help='Train dir')
parse.add_argument('--test',type=str, default='images/test',help='Test dir')
parse.add_argument('--split',type=int, default=80, help='Train/Test ratio \n split=20 --> train/test=80/20')
parse.add_argument('--in_shape',type=tuple, default=(128,128),help='Input shape to model')
parse.add_argument('--batch',type=int, default=32, help='Batch size to train')
parse.add_argument('--epochs',type=int, default=10, help='Epochs to train')
parse.add_argument('--seed',type=int,default=42,help='Random seed')

args = parse.parse_args()

train = tf.keras.utils.image_dataset_from_directory(args.train,
                                                    validation_split=args.split/100,
                                                    subset='training',
                                                    seed=args.seed,
                                                    image_size=args.in_shape,
                                                    batch_size = args.batch)
val = tf.keras.utils.image_dataset_from_directory(args.train,
                                                    validation_split=args.split/100,
                                                    subset='validation',
                                                    seed=args.seed,
                                                    image_size=args.in_shape,
                                                    batch_size = args.batch)

print(train.class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = Sequential([
    layers.RandomRotation(factor=(-0.2,0.2),seed = 123),
    layers.RandomZoom(0.1),
    layers.RandomFlip('horizontal'),
])

base_model = tf.keras.applications.vgg16.VGG16(weights='imagenet',
                                               include_top=False,
                                            #    input_shape=(256,256,3),
                                                )

model = Sequential([
    layers.Rescaling(1./255.,input_shape=args.in_shape+(3,)),
    data_augmentation,
    base_model,
    layers.Flatten(),
    # layers.Dense(4096,activation='relu'),
    layers.Dense(1024,activation='relu'),
    layers.Dense(256,activation='relu'),
    layers.Dense(4,activation='softmax'),   
])
model.build()
print(model.summary())
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
history = model.fit(train_ds,
          validation_data=val_ds,
          epochs=args.epochs)

# loss, acc = model.evaluate()
# print('Acc: {acc}, Loss: {loss}')
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(args.epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()