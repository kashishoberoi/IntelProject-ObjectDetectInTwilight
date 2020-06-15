import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import glob
from PIL import Image 
import numpy as np
from sklearn.model_selection import train_test_split
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
batch_size = 16
epochs = 10
IMG_HEIGHT = 416
IMG_WIDTH = 416
total_train = 787
total_val = 338
train_dir = ".\\myweatherdata\\train"
validation_dir = ".\\myweatherdata\\test"
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

with tf.device('/GPU:0'):
    train_image_generator = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )
    validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,directory=train_dir, shuffle=True, target_size=(IMG_HEIGHT, IMG_WIDTH))
    sample_training_images, _ = next(train_data_gen)
    val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,directory=validation_dir,target_size=(IMG_HEIGHT, IMG_WIDTH))
    sample_training_images, _ = next(train_data_gen)
    print("Building Model...")
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), strides = [1,1],padding = [[0,0],[1,1],[1,1],[0,0]] ,activation='relu', input_shape=(416, 416, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), strides = [1,1],padding = [[0,0],[1,1],[1,1],[0,0]] ,activation='relu'))
    model.add(layers.Conv2D(32, (3, 3), strides = [1,1],padding = [[0,0],[1,1],[1,1],[0,0]] ,activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), strides = [1,1],padding = [[0,0],[1,1],[1,1],[0,0]] ,activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), strides = [1,1],padding = [[0,0],[1,1],[1,1],[0,0]] ,activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), strides = [1,1],padding = [[0,0],[1,1],[1,1],[0,0]] ,activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), strides = [1,1],padding = [[0,0],[1,1],[1,1],[0,0]] ,activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), strides = [1,1],padding = [[0,0],[1,1],[1,1],[0,0]] ,activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), strides = [1,1],padding = [[0,0],[1,1],[1,1],[0,0]] ,activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), strides = [1,1],padding = [[0,0],[1,1],[1,1],[0,0]] ,activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(4))
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=total_train // batch_size,
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=total_val // batch_size
    )
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss=history.history['loss']
    val_loss=history.history['val_loss']

    epochs_range = range(epochs)

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