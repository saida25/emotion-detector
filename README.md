🚀 A webcam-based **Emotion Detector** using **OpenCV + TensorFlow (CNN)** is a great hands-on computer vision project.

### Set Up Your Environment

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

pip install tensorflow opencv-python keras numpy matplotlib
```

---

### Dataset

Use **FER2013 dataset** (common for emotion recognition). Download from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013).
Save dataset in `data/` folder.

---

### Code Structure

```
emotion-detector/
│── data/                # dataset (FER2013 or custom)
│── models/              # trained CNN models
│── app.py               # main webcam app
│── train.py             # CNN training script
│── requirements.txt     # dependencies
│── README.md            # project documentation
```



# Emotion Detector 🎭

A real-time **Emotion Detector** using **OpenCV** and a **CNN (TensorFlow/Keras)** model trained on the FER2013 dataset.

## 🚀 Features
- Detects emotions (Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral)
- Real-time webcam feed
- CNN trained with FER2013 dataset

## 🛠️ Tech Stack
- Python
- TensorFlow/Keras
- OpenCV
- NumPy

## 📂 Project Structure
```

emotion-detector/
│── data/                # dataset
│── models/              # trained models
│── app.py               # main webcam app
│── train.py             # CNN training
│── requirements.txt     # dependencies
│── README.md            # docs

````

## 📦 Installation
```bash
git clone https://github.com/your-username/emotion-detector.git
cd emotion-detector
pip install -r requirements.txt
````

## 🔧 Usage

1. Train the model:

   ```bash
   python train.py
   ```

   (or download pretrained model and put inside `models/`)

2. Run the app:

   ```bash
   python app.py
   ```

3. Press `q` to quit the webcam.

## 📊 Dataset

Uses the [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013).

## 🙌 Acknowledgments

* OpenCV for real-time face detection
* TensorFlow/Keras for deep learning
* Kaggle FER2013 dataset


