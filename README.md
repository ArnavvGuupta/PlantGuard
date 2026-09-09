# 🌿 PlantGuard — AI-Powered Plant Disease Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/CNN-Computer%20Vision-8A2BE2?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
</p>

<p align="center">
  <b>🌱 An AI-powered computer vision system for detecting plant diseases from leaf images.</b>
</p>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=18&duration=3000&pause=1000&color=58A6FF&center=true&vCenter=true&width=600&lines=Upload+a+leaf+image+%F0%9F%8D%83;Let+the+CNN+analyze+it+%F0%9F%A7%A0;Detect+plant+disease+%F0%9F%8C%BF;Fast+%26+interactive+predictions+%F0%9F%9A%80" />
</p>

---

## 📌 Overview

**PlantGuard** is a machine-learning application that uses **Convolutional Neural Networks (CNNs)** to identify plant diseases from leaf images.

The system was trained on **10,849 labelled leaf images across 38 disease classes** and achieved approximately **91% test accuracy**.

Users can upload an image through the Streamlit interface, after which the trained model analyzes the image and returns the predicted disease class.

---

## ✨ Features

* 🌿 **Plant disease classification**
* 🧠 CNN-based image classification
* 📊 **38 disease classes**
* 🖼️ Drag-and-drop image upload
* ⚡ Fast inference
* 🎯 ~**91% test accuracy**
* 💻 Interactive Streamlit interface
* 🔬 TensorFlow-based ML pipeline
* 📦 Production-trained `.h5` model

---

## 🧠 Machine Learning Pipeline

```text
             ┌──────────────────┐
             │   Leaf Image 📷  │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ Image Processing │
             │ & Normalization  │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │       CNN        │
             │  Image Features  │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ Disease          │
             │ Classification   │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ Prediction 🌱    │
             └──────────────────┘
```

---

## 📊 Model Performance

| Metric          |         Result |
| --------------- | -------------: |
| Training Images |     **10,849** |
| Disease Classes |         **38** |
| Model           |        **CNN** |
| Framework       | **TensorFlow** |
| Test Accuracy   |       **~91%** |
| Inference Time  | **<2 seconds** |

> Performance can vary depending on image quality, lighting, background, and similarity between diseases.

---

## 🗂️ Project Structure

```text
PlantGuard/
│
├── app/
│   ├── models/
│   │   └── plant_disease_production_model.h5
│   │
│   └── ...
│
├── notebooks/
│   └── ...
│
├── requirements.txt
├── .gitignore
├── README.md
└── ...
```

---

## 🛠️ Tech Stack

### Machine Learning

* Python
* TensorFlow
* CNN
* NumPy
* Pandas
* Scikit-learn

### Application

* Streamlit
* Python

### Development

* Git
* GitHub
* Jupyter Notebook
* VS Code / PyCharm

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/ArnavvGuupta/PlantGuard.git
cd PlantGuard
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🤖 Model Setup

The trained `.h5` model is **not included in this Git repository** because the model file exceeds GitHub's standard **100 MB file-size limit**.

Expected model location:

```text
app/models/plant_disease_production_model.h5
```

After obtaining the trained model, place it inside:

```text
app/models/
```

The application will then load the model for inference.

---

## 🚀 Run the Application

Start the Streamlit application with:

```bash
streamlit run app.py
```

The application will open locally in your browser.

You can then:

```text
Upload Leaf Image
        ↓
Image Preprocessing
        ↓
CNN Prediction
        ↓
Disease Classification
```

---

## 🖥️ Application Workflow

### 1️⃣ Upload

Upload a clear image of a plant leaf.

### 2️⃣ Process

The application preprocesses the image into the format expected by the trained CNN.

### 3️⃣ Predict

The CNN extracts visual features and predicts the most likely disease class.

### 4️⃣ Result

The predicted disease is displayed through the Streamlit interface.

---

## 🔬 Why CNN?

Convolutional Neural Networks are particularly effective for image classification because they can automatically learn visual patterns such as:

* Leaf texture
* Color variations
* Spots and lesions
* Disease patterns
* Shape abnormalities
* Edge and structural features

Instead of manually defining these characteristics, the CNN learns useful representations directly from the training images.

---

## 🎯 Future Improvements

* [ ] Add confidence scores for predictions
* [ ] Add disease treatment recommendations
* [ ] Support more plant species
* [ ] Improve model performance with transfer learning
* [ ] Add image augmentation pipelines
* [ ] Deploy the model using a cloud inference API
* [ ] Add mobile-friendly UI
* [ ] Add explainable AI using Grad-CAM
* [ ] Containerize the application with Docker

---

## ⚠️ Limitations

PlantGuard is an AI-based classification system and should be treated as a **decision-support tool**, not a replacement for professional agricultural diagnosis.

Prediction quality can be affected by:

* Poor image quality
* Low lighting
* Occluded leaves
* Multiple diseases on the same leaf
* Unseen plant species
* Background noise

---

## 📸 Demo

> Add screenshots or a short GIF of the Streamlit application here.

```text
┌─────────────────────────────────────┐
│          🌿 PlantGuard              │
│                                     │
│   Upload a leaf image               │
│   ┌─────────────────────────────┐   │
│   │     Drag & Drop Image       │   │
│   └─────────────────────────────┘   │
│                                     │
│   Prediction:                       │
│   🧠 Disease Class                  │
│                                     │
└─────────────────────────────────────┘
```

---

## 👨‍💻 Author

### **Arnav Gupta**

Final-year B.Tech — Electrical & Electronics Engineering
Maharaja Agrasen Institute of Technology (MAIT), GGSIPU

Interested in:

**AI/ML • Generative AI • Cloud Engineering • Full-Stack Development**

<p>
  <a href="https://github.com/ArnavvGuupta">
    <img src="https://img.shields.io/badge/GitHub-ArnavvGuupta-181717?style=for-the-badge&logo=github&logoColor=white" />
  </a>
  <a href="https://linkedin.com/in/arnavgupta-281886240">
    <img src="https://img.shields.io/badge/LinkedIn-Arnav%20Gupta-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" />
  </a>
</p>

---

<div align="center">

### 🌱 PlantGuard

**Turning computer vision into practical agricultural intelligence.**

⭐ If you found this project interesting, consider giving it a star!

</div>
