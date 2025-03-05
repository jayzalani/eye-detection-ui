import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import cv2
import numpy as np
from model.ipynb 
import plot_actual_vs_predicted

def augment_data(train_df, valid_df, test_df, batch_size=16):
    img_size = (256, 256)
    channels = 3
    color = 'rgb'

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=30,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.5, 1.5])

    valid_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='filepaths',
        y_col='labels',
        target_size=img_size,
        color_mode=color,
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical'
    )

    print("Shape of augmented training images:", train_generator.image_shape)

    valid_generator = valid_test_datagen.flow_from_dataframe(
        valid_df,
        x_col='filepaths',
        y_col='labels',
        target_size=img_size,
        color_mode=color,
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical'
    )

    print("Shape of validation images:", valid_generator.image_shape)

    test_generator = valid_test_datagen.flow_from_dataframe(
        test_df,
        x_col='filepaths',
        y_col='labels',
        target_size=img_size,
        color_mode=color,
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical'
    )

    print("Shape of test images:", test_generator.image_shape)

    return train_generator, valid_generator, test_generator

def load_model(num_classes):
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    predictions = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

"""def predict_disease(image_path, model):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    class_labels = list(train_generator.class_indices.keys())
    predicted_label = class_labels[np.argmax(prediction)]
    return predicted_label"""
def plot_actual_vs_predicted():
    return 0