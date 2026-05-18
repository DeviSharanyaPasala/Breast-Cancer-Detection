import os
import numpy as np
import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

DATASET_PATH = os.environ.get("DATASET_PATH", "data")
MODEL_OUTPUT = os.environ.get("MODEL_OUTPUT", "model/breast_cancer_model.h5")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = int(os.environ.get("EPOCHS", "10"))

os.makedirs("model", exist_ok=True)

def build_model():
    base = DenseNet121(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base.input, outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train():
    mlflow.set_experiment("breast-cancer-detection")
    with mlflow.start_run():
        mlflow.log_param("model", "DenseNet121")
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("img_size", IMG_SIZE)

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            horizontal_flip=True,
            validation_split=0.2
        )

        train_gen = train_datagen.flow_from_directory(
            DATASET_PATH,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="binary",
            subset="training"
        )

        val_gen = train_datagen.flow_from_directory(
            DATASET_PATH,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="binary",
            subset="validation"
        )

        model = build_model()
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS
        )

        val_loss = min(history.history["val_loss"])
        val_acc = max(history.history["val_accuracy"])

        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_metric("val_loss", val_loss)
        mlflow.keras.log_model(model, "model")

        model.save(MODEL_OUTPUT)
        print(f"Model saved to {MODEL_OUTPUT}")
        print(f"Val accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    train()
