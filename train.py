import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
import tensorflow as tf

DATASET_PATH = "data/genres"

# ✅ Only keep real directories
GENRES = sorted([
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
])

X = []
y = []

print("Loading dataset...")

for label, genre in enumerate(GENRES):
    genre_path = os.path.join(DATASET_PATH, genre)

    # ✅ Only process audio files
    for file in os.listdir(genre_path):
        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(genre_path, file)

        try:
            y_audio, sr = librosa.load(file_path, duration=30)

            mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=40)
            mfcc = np.mean(mfcc.T, axis=0)

            X.append(mfcc)
            y.append(label)

        except Exception as e:
            print("Skipping:", file, "| Error:", e)
            continue

X = np.array(X)
y = tf.keras.utils.to_categorical(y, num_classes=len(GENRES))

print("Dataset shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(40,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(GENRES), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=20, batch_size=32)

model.save("model.h5")

print("✅ Model saved as model.h5")