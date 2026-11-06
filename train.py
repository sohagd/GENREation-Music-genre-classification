"""Train and evaluate a CNN on the GTZAN genre dataset."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.audio_features import extract_mfcc, MAX_FRAMES, N_MFCC

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]

def discover_audio_files(dataset_dir):
    root = Path(dataset_dir)
    records = []
    # GTZAN Kaggle mirrors commonly use genres_original/<genre>/*.wav.
    for genre in GENRES:
        folder = root / genre
        if not folder.exists():
            folder = root / "genres_original" / genre
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*")):
            if path.suffix.lower() in {".wav", ".mp3", ".au", ".flac", ".ogg"}:
                records.append((path, genre))
    return records

def build_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/genres_original",
                        help="Folder containing one subfolder per genre, or the GTZAN root.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs("artifacts", exist_ok=True)
    tf.keras.utils.set_random_seed(args.seed)
    records = discover_audio_files(args.data_dir)
    if not records:
        raise SystemExit(
            f"No audio files found under {args.data_dir}. Expected genre folders such as "
            "'blues', 'classical', ...; see README for dataset setup."
        )

    print(f"Found {len(records)} audio files. Extracting MFCC features...")
    X, y, paths = [], [], []
    for i, (path, genre) in enumerate(records, start=1):
        try:
            X.append(extract_mfcc(path))
            y.append(GENRES.index(genre))
            paths.append(str(path))
        except Exception as exc:
            print(f"[SKIP] {path}: {exc}")
        if i % 50 == 0:
            print(f"Processed {i}/{len(records)}")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if len(np.unique(y)) < 2:
        raise SystemExit("Need audio from at least two genre folders to train a classifier.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=args.seed, stratify=y
    )
    print(f"Train samples: {len(X_train)} | Validation samples: {len(X_val)}")
    model = build_model((N_MFCC, MAX_FRAMES, 1), len(GENRES))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                         restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("artifacts/music_genre_cnn.keras",
                                           monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=2, min_lr=1e-6),
    ]
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks, verbose=1
    )

    model = tf.keras.models.load_model("artifacts/music_genre_cnn.keras")
    probs = model.predict(X_val, batch_size=args.batch_size, verbose=0)
    preds = probs.argmax(axis=1)
    accuracy = float(accuracy_score(y_val, preds))
    report = classification_report(y_val, preds, labels=list(range(len(GENRES))),
                                   target_names=GENRES, output_dict=True, zero_division=0)
    print(f"\nValidation accuracy: {accuracy:.4f}")
    print(classification_report(y_val, preds, labels=list(range(len(GENRES))),
                               target_names=GENRES, zero_division=0))

    with open("artifacts/labels.json", "w", encoding="utf-8") as f:
        json.dump({"genres": GENRES}, f, indent=2)
    with open("artifacts/evaluation.json", "w", encoding="utf-8") as f:
        json.dump({"validation_accuracy": accuracy, "classification_report": report}, f, indent=2)

    cm = confusion_matrix(y_val, preds, labels=list(range(len(GENRES))))
    plt.figure(figsize=(11, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=GENRES, yticklabels=GENRES)
    plt.title("GTZAN Music Genre Classification — Confusion Matrix")
    plt.xlabel("Predicted genre")
    plt.ylabel("Actual genre")
    plt.tight_layout()
    plt.savefig("artifacts/confusion_matrix.png", dpi=160)
    plt.close()

    pd.DataFrame(history.history).to_csv("artifacts/training_history.csv", index=False)
    plt.figure(figsize=(9, 4))
    plt.plot(history.history["accuracy"], label="Training accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig("artifacts/training_history.png", dpi=160)
    plt.close()
    print("Saved model and evaluation files in ./artifacts")

if __name__ == "__main__":
    main()