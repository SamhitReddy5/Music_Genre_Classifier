# Music Genre Classifier

A machine learning web application that predicts the genre of an audio file using MFCC feature extraction and a neural network model.

---

## Features

- Upload `.wav` audio files  
- Predict music genre using a trained model  
- Built with Flask + TensorFlow/Keras  
- Simple and lightweight web interface  

---

## Screenshots

### Home Page
![Home](screenshots/home.png)

### Upload
![Upload](screenshots/upload.png)

### Result
![Result](screenshots/result.png)

---

## Project Structure

```
music-genre-classifier/
├── app.py              # Flask web app
├── predict.py          # Prediction logic
├── train.py            # Model training script
├── requirements.txt    # Dependencies
├── samples/            # Sample audio files
├── screenshots/        # UI screenshots
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Music_Genre_Classifier.git
cd Music_Genre_Classifier
```

Create virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## Training the Model

```bash
python train.py
```

This generates:

```
model.h5
```

---

## Notes

- Only `.wav` files are supported  
- Dataset and trained model are not included due to size  
- This is a basic neural network (not production-grade)

---

## Future Improvements

- Add MP3 support  
- Display prediction confidence  
- Improve UI  
- Upgrade to CNN model  
- Deploy online  

---

## Author

Samhit Reddy  
https://github.com/SamhitReddy5
