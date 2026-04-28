import numpy as np
import librosa
import tensorflow as tf

# Load model once
model = tf.keras.models.load_model("model.h5", compile=False)

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)

    return mfcc.reshape(1, 40)


def predict_genre(file_path):
    features = extract_features(file_path)
    pred = model.predict(features, verbose=0)

    return GENRES[int(np.argmax(pred))]