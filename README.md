The project contains the following key files and directories:

# ML-Project-Sem4: Brain Tumor Detection Using VGG16

Welcome to the **ML-Project-Sem4** repository! This project aims to detect brain tumors in MRI images using deep learning, specifically leveraging the VGG16 architecture to achieve high accuracy (98%).

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

Brain tumor detection is a crucial application of AI in the medical field. This project implements a Convolutional Neural Network (CNN) based on the **VGG16** architecture to classify MRI images as either having a tumor or being tumor-free.

**Key Features:**
- Utilizes transfer learning for efficient model training.
- Achieves high accuracy (~98%) on a public dataset.
- Includes well-documented Jupyter notebooks for reproducibility.

---

## Dataset

The dataset is located in the `brain_tumor_dataset` folder and contains two subdirectories:
- `yes`: Images with brain tumors.
- `no`: Images without brain tumors.

Ensure the dataset is structured correctly before training the model:
```
brain_tumor_dataset/
├── yes/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── no/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

---

## Project Structure

```
ML-Project-Sem4/
├── 2_detecting-brain-tumors-vgg16-accuracy-98.ipynb   # Model training and evaluation notebook
├── detecting-brain-tumors-vgg16-accuracy-98.ipynb    # Alternate notebook with experimentation
├── brain_tumor_dataset/                              # Dataset directory
│   ├── yes/                                          # Tumor-positive images
│   ├── no/                                           # Tumor-negative images
├── README.md                                         # Existing README file
├── .gitignore                                        # Files ignored by version control
└── .git                                              # Git repository metadata
```

---

## Installation

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Virtual environment (optional)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/AnouskaJ/ML-Project-Sem4.git
   cd ML-Project-Sem4
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Open the relevant notebook (`2_detecting-brain-tumors-vgg16-accuracy-98.ipynb`) and run the cells.

---

## Usage

### Training the Model

1. Load the dataset into the notebook.
2. Preprocess the images:
   - Resize all images to the required input size (e.g., 224x224 for VGG16).
3. Fine-tune the VGG16 model for the binary classification task.
4. Train the model on the provided dataset.

### Inference

Use the trained model to classify new MRI images. For example:
```python
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Load the model
model = load_model('path_to_saved_model.h5')

# Load and preprocess an image
image = load_img('path_to_image.jpg', target_size=(224, 224))
image = img_to_array(image) / 255.0
image = np.expand_dims(image, axis=0)

# Predict
prediction = model.predict(image)
print("Tumor Detected" if prediction[0] > 0.5 else "No Tumor")
```

---

## Results

- **Accuracy**: Achieved 98% accuracy on the test dataset.
- **Precision and Recall**: [Add details if available]
- **Loss and Accuracy Curves**: Visualized in the notebooks.

---

## Contributing

We welcome contributions to improve the project! To contribute:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Description of changes"
   ```
4. Push the branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Submit a pull request.

---
