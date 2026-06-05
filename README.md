<div align="center">

<!-- Animated Face Scanner SVG -->
<svg xmlns="http://www.w3.org/2000/svg" width="160" height="160" viewBox="0 0 160 160">
  <defs>
    <!-- Scanning glow gradient -->
    <linearGradient id="scanGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#00ffcc" stop-opacity="0"/>
      <stop offset="50%" stop-color="#00ffcc" stop-opacity="0.4"/>
      <stop offset="100%" stop-color="#00ffcc" stop-opacity="0"/>
    </linearGradient>
  </defs>

  <!-- Circular Radar Rings -->
  <circle cx="80" cy="80" r="72" fill="none" stroke="#00ffcc" stroke-width="0.5" stroke-opacity="0.15"/>
  <circle cx="80" cy="80" r="50" fill="none" stroke="#00ffcc" stroke-width="0.5" stroke-opacity="0.15"/>

  <!-- Corner Guides (Camera Bracket) -->
  <path d="M 25,45 L 25,25 L 45,25" fill="none" stroke="#00ffcc" stroke-width="1.5" stroke-linecap="round"/>
  <path d="M 135,45 L 135,25 L 115,25" fill="none" stroke="#00ffcc" stroke-width="1.5" stroke-linecap="round"/>
  <path d="M 25,115 L 25,135 L 45,135" fill="none" stroke="#00ffcc" stroke-width="1.5" stroke-linecap="round"/>
  <path d="M 135,115 L 135,135 L 115,135" fill="none" stroke="#00ffcc" stroke-width="1.5" stroke-linecap="round"/>

  <!-- Face Wireframe Silhouette -->
  <!-- Head outline -->
  <path d="M 50,55 Q 80,42 110,55 Q 115,85 105,108 Q 80,128 55,108 Q 45,85 50,55 Z" 
        fill="none" stroke="#0080ff" stroke-width="1.2" stroke-opacity="0.5"/>
  <!-- Eyes -->
  <circle cx="68" cy="72" r="3" fill="#00ffcc" opacity="0.8">
    <animate attributeName="opacity" values="0.3;1;0.3" dur="2s" repeatCount="indefinite"/>
  </circle>
  <circle cx="92" cy="72" r="3" fill="#00ffcc" opacity="0.8">
    <animate attributeName="opacity" values="0.3;1;0.3" dur="2s" repeatCount="indefinite"/>
  </circle>
  <!-- Nose/mouth path -->
  <line x1="80" y1="72" x2="80" y2="88" stroke="#0080ff" stroke-width="1" stroke-opacity="0.6"/>
  <path d="M 70,98 Q 80,108 90,98" fill="none" stroke="#00ffcc" stroke-width="1.5" stroke-linecap="round">
    <animate attributeName="stroke" values="#00ffcc;#ff3366;#00ffcc" dur="4s" repeatCount="indefinite"/>
  </path>

  <!-- Horizontal Scanning Line -->
  <g>
    <rect x="20" y="20" width="120" height="20" fill="url(#scanGrad)"/>
    <line x1="20" y1="30" x2="140" y2="30" stroke="#00ffcc" stroke-width="1.5">
      <animate attributeName="stroke-opacity" values="0.6;1;0.6" dur="1s" repeatCount="indefinite"/>
    </line>
    <animateTransform attributeName="transform" type="translate"
      values="0,0; 0,100; 0,0" dur="3s" repeatCount="indefinite"/>
  </g>

  <!-- Blinking Landmark Nodes -->
  <circle cx="80" cy="50" r="1.5" fill="#ff3366">
    <animate attributeName="opacity" values="0.2;1;0.2" dur="1.5s" repeatCount="indefinite"/>
  </circle>
  <circle cx="50" cy="80" r="1.5" fill="#ff3366">
    <animate attributeName="opacity" values="1;0.2;1" dur="1.5s" repeatCount="indefinite"/>
  </circle>
  <circle cx="110" cy="80" r="1.5" fill="#ff3366">
    <animate attributeName="opacity" values="0.2;1;0.2" dur="1.8s" repeatCount="indefinite"/>
  </circle>
  <circle cx="80" cy="115" r="1.5" fill="#ff3366">
    <animate attributeName="opacity" values="1;0.2;1" dur="1.2s" repeatCount="indefinite"/>
  </circle>
</svg>

<br/>
<img src="https://readme-typing-svg.demolab.com?font=Orbitron&weight=900&size=32&duration=3000&pause=1000&color=00FFE7&center=true&vCenter=true&width=600&lines=Facial+Emotion+Recognition;Real-Time+Webcam+CNN;Inference+%26+Visualization;FER-2013+Model" alt="Facial Emotion Recognition" />
</div>

---

<div align="center">

<h3><i>Real-Time Deep Learning Emotion Classifier</i></h3>

<!-- Shield Badges -->
![Python](https://img.shields.io/badge/Python-3.11-00ffe7?style=for-the-badge&logo=python&logoColor=black&labelColor=060f1e)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-ff6f00?style=for-the-badge&logo=tensorflow&logoColor=white&labelColor=060f1e)
![Keras 3](https://img.shields.io/badge/Keras-3-D00000?style=for-the-badge&logo=keras&logoColor=white&labelColor=060f1e)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white&labelColor=060f1e)
![License](https://img.shields.io/badge/License-MIT-f5a623?style=for-the-badge&labelColor=060f1e)

![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-060f1e?style=for-the-badge&labelColor=333333)

<br/>

**Advanced Convolutional Neural Network (CNN) for real-time facial expression and emotion classification from webcam and static images.**

<br/>

🚀 [Setup Guide](#-step-by-step-setup--workflow-guide) • 🧬 [CNN Architecture](#-model-architecture) • 📈 [Performance](#-performance--tuning) • 🎭 [Emotion Map](#-emotion-classification-mapping)

</div>

---

## 📁 Project Structure

```
Facial-Emotion/
├── run_test.bat                     ← Start webcam testing (Windows shortcut)
├── run_image_test.bat               ← Start image testing (Windows shortcut)
├── run_test.sh                      ← Start webcam testing (Linux/macOS shortcut)
├── run_image_test.sh                ← Start image testing (Linux/macOS shortcut)
├── requirements.txt                 ← Python dependencies
├── .gitignore                       ← Excludes models and large datasets
├── main.py                          ← CNN model training and augmentation
├── test.py                          ← Real-time webcam detector
├── testdata.py                      ← Single static image analyzer
├── haarcascade_frontalface_default.xml  ← Haar Cascade face detector
└── Data/                            ← Local dataset folder (Excluded from Git)
    ├── train/                       ← Training subset sorted by emotions
    └── test/                        ← Validation subset sorted by emotions
```

---

## ⚙️ Step-by-Step Setup & Workflow Guide

Follow these steps to configure your environment, train the CNN model, and run emotion recognition inference:

### 1 · Clone the Repository
Clone this repository to your local machine and navigate into the project root:
```bash
git clone https://github.com/ChiranjibSaiChandanNath/Facial-Emotion-Recognization.git
cd Facial-Emotion-Recognization
```

### 2 · Create a Virtual Environment
Initialize a local Python virtual environment to manage dependencies isolated from your system packages:

#### 🪟 Windows
```powershell
# Create environment
python -m venv venv

# Activate (Command Prompt)
venv\Scripts\activate

# Activate (PowerShell)
venv\Scripts\Activate.ps1
```

#### 🍎 macOS
```bash
# Create environment
python3 -m venv venv

# Activate
source venv/bin/activate
```

#### 🐧 Linux
```bash
# Create environment
python3 -m venv venv

# Activate
source venv/bin/activate
```

### 3 · Install Project Dependencies
Use `pip` to install the required deep learning, image processing, and numerical packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4 · Prepare the Dataset Structure
The dataset must be placed in a directory named `Data/` at the root of the project. 

> [!TIP]
> You can download the prepared dataset directly from **[Google Drive Dataset Link](https://drive.google.com/file/d/1oV8grnQY5m_slj_VybKI_SAdzHZLZsYS/view?usp=sharing)**. Once downloaded, extract and organize your training and testing images as follows:
```
Data/
├── train/
│   ├── Angry/
│   ├── Disgust/
│   ├── Fear/
│   ├── Happy/
│   ├── Neutral/
│   ├── Sad/
│   └── Surprise/
└── test/
    ├── Angry/
    ├── Disgust/
    └── ... (same subfolders for validation)
```

### 5 · Train the CNN & Generate `model_file.h5`
To train the Convolutional Neural Network on the images inside `Data/`:
```bash
python main.py
```
**What this script does:**
1. Loads training images from `Data/train` and validation images from `Data/test`.
2. Applies real-time data augmentation (rotations, zooms, horizontal flips) to expand and generalize the training dataset.
3. Builds the CNN structure (Conv2D -> MaxPooling -> Dropout -> Flatten -> Dense).
4. Fits the model, monitoring validation accuracy, and saves the best model states automatically.
5. Saves the final trained model weights as `model_file.h5` in the root folder.

### 6 · Run Facial Emotion Inference
To run inference, you need the trained model weights:
- **Option A (Pre-trained)**: Download the pre-trained **[model_file.h5 Weights](YOUR_GOOGLE_DRIVE_MODEL_LINK_HERE)** directly and place it in the root folder of the project.
- **Option B (Train from scratch)**: Run the training step (**Step 5** above) to generate `model_file.h5` yourself.

Once `model_file.h5` is in your root directory, run the classification scripts matching your operating system:

#### 🪟 Windows
- **Real-Time Webcam Mode**: Double-click `run_test.bat` (or run `python test.py`).
- **Static Image Mode**: Double-click `run_image_test.bat` to test on `img2.jpeg` (or run `python testdata.py path_to_image.jpg`).

#### 🍎 macOS
First, grant execution permissions to the launcher scripts:
```bash
chmod +x run_test.sh run_image_test.sh
```
- **Real-Time Webcam Mode**: Run `./run_test.sh` (or run `python3 test.py`).
- **Static Image Mode**: Run `./run_image_test.sh path_to_image.jpg` (or run `python3 testdata.py path_to_image.jpg`).

#### 🐧 Linux
First, grant execution permissions to the launcher scripts:
```bash
chmod +x run_test.sh run_image_test.sh
```
- **Real-Time Webcam Mode**: Run `./run_test.sh` (or run `python3 test.py`).
- **Static Image Mode**: Run `./run_image_test.sh path_to_image.jpg` (or run `python3 testdata.py path_to_image.jpg`).

---

## 🧬 Model Architecture

```
1. Input Layer: Grayscale Images (48 x 48 x 1)
2. Conv2D (32 filters, 3x3 kernel, ReLU)
3. Conv2D (64 filters, 3x3 kernel, ReLU) -> MaxPooling2D -> Dropout(10%)
4. Conv2D (128 filters, 3x3 kernel, ReLU) -> MaxPooling2D -> Dropout(10%)
5. Conv2D (256 filters, 3x3 kernel, ReLU) -> MaxPooling2D -> Dropout(10%)
6. Flatten Layer
7. Fully Connected Dense Layer (512 units, ReLU) -> Dropout(20%)
8. Output Classification Layer (7 units, Softmax)
```

---

## 🎭 Emotion Classification Mapping

<!-- Animated confidence scale SVG -->
<div align="center">
<svg xmlns="http://www.w3.org/2000/svg" width="500" height="36" viewBox="0 0 500 36">
  <defs>
    <linearGradient id="emotionGrad" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%"   stop-color="#00ffe7"/>
      <stop offset="20%"  stop-color="#00ff88"/>
      <stop offset="40%"  stop-color="#f5a623"/>
      <stop offset="60%"  stop-color="#ff7c2a"/>
      <stop offset="80%"  stop-color="#ff3c6e"/>
      <stop offset="100%" stop-color="#ff00ff"/>
    </linearGradient>
  </defs>
  <!-- Track -->
  <rect x="10" y="14" width="480" height="8" rx="4" fill="rgba(255,255,255,0.05)" stroke="rgba(255,255,255,0.1)" stroke-width="0.5"/>
  <!-- Animated fill -->
  <rect x="10" y="14" width="0" height="8" rx="4" fill="url(#emotionGrad)">
    <animate attributeName="width" from="0" to="480" dur="2s" begin="0.5s" fill="freeze"/>
  </rect>
  <!-- Labels -->
  <text x="10"  y="9" font-family="monospace" font-size="7" fill="#00ffe7">NEUTRAL</text>
  <text x="105" y="9" font-family="monospace" font-size="7" fill="#00ff88" text-anchor="middle">HAPPY</text>
  <text x="200" y="9" font-family="monospace" font-size="7" fill="#f5a623" text-anchor="middle">SURPRISE</text>
  <text x="295" y="9" font-family="monospace" font-size="7" fill="#ff7c2a" text-anchor="middle">FEAR / SAD</text>
  <text x="480" y="9" font-family="monospace" font-size="7" fill="#ff3c6e" text-anchor="end">ANGRY / DISGUST</text>
  <!-- Tick marks -->
  <line x1="105" y1="12" x2="105" y2="24" stroke="#00ff88" stroke-width="0.8" stroke-opacity="0.6"/>
  <line x1="200" y1="12" x2="200" y2="24" stroke="#f5a623" stroke-width="0.8" stroke-opacity="0.6"/>
  <line x1="295" y1="12" x2="295" y2="24" stroke="#ff7c2a" stroke-width="0.8" stroke-opacity="0.6"/>
</svg>
</div>

<br/>

The system detects facial landmarks and draws visual color boxes specific to each emotion:

| Index | Emotion | Bounding Box Color | Color Representation (RGB) |
|:-----:|:-------:|:------------------:|:--------------------------:|
| **0** | 🔴 Angry | Red | `(0, 0, 255)` |
| **1** | 🟠 Disgust | Orange | `(0, 140, 255)` |
| **2** | 🟡 Fear | Yellow | `(0, 255, 255)` |
| **3** | 🟢 Happy | Green | `(0, 255, 0)` |
| **4** | ⚪ Neutral | White | `(255, 255, 255)` |
| **5** | 🔵 Sad | Blue | `(255, 0, 0)` |
| **6** | 🟣 Surprise | Magenta | `(255, 0, 255)` |

---

## 📈 Performance & Tuning

- **Grayscale Normalization**: Pixel values are rescaled `1./255` prior to input.
- **Data Augmentation**: Shear, zoom, rotation, and flips are applied during training to prevent overfitting.
- **Dynamic Checkpoint**: Callback saves the best performing weights to `model_file.h5` based on validation accuracy (`ModelCheckpoint`).

---

<div align="center">

<!-- Animated bottom banner SVG -->
<svg xmlns="http://www.w3.org/2000/svg" width="600" height="48" viewBox="0 0 600 48">
  <defs>
    <linearGradient id="bannerGrad" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#00ffe7" stop-opacity="0"/>
      <stop offset="25%" stop-color="#00ffe7" stop-opacity="0.6"/>
      <stop offset="75%" stop-color="#00ffe7" stop-opacity="0.6"/>
      <stop offset="100%" stop-color="#00ffe7" stop-opacity="0"/>
    </linearGradient>
  </defs>
  <!-- Top line -->
  <rect x="0" y="0" width="600" height="1" fill="url(#bannerGrad)"/>
  <!-- Bottom line -->
  <rect x="0" y="47" width="600" height="1" fill="url(#bannerGrad)"/>
  <!-- Text -->
  <text x="300" y="20" font-family="'Courier New', monospace" font-size="11"
    fill="#00ffe7" text-anchor="middle" letter-spacing="4" opacity="0.9">
    EMOTION RECOGNITION 2026
    <animate attributeName="opacity" values="0.5;1;0.5" dur="3s" repeatCount="indefinite"/>
  </text>
  <text x="300" y="38" font-family="'Courier New', monospace" font-size="9"
    fill="#4a6a8a" text-anchor="middle" letter-spacing="3">
    COMPUTER VISION &amp; DEEP LEARNING
  </text>
  <!-- Corner accents -->
  <path d="M8,4 L4,4 L4,14" fill="none" stroke="#00ffe7" stroke-width="1" opacity="0.6"/>
  <path d="M592,4 L596,4 L596,14" fill="none" stroke="#00ffe7" stroke-width="1" opacity="0.6"/>
  <path d="M8,44 L4,44 L4,34" fill="none" stroke="#00ffe7" stroke-width="1" opacity="0.6"/>
  <path d="M592,44 L596,44 L596,34" fill="none" stroke="#00ffe7" stroke-width="1" opacity="0.6"/>
</svg>

<br/><br/>

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=facial-emotion.recognition&left_color=060f1e&right_color=00ffe7&left_text=Visitors)

</div>
