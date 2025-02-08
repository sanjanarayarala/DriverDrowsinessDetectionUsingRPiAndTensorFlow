# Driver Drowsiness Detection Using Raspberry Pi and TensorFlow

This project uses a **Raspberry Pi** to detect drowsiness in a driver by monitoring their eye state (open or closed) using a **Convolutional Neural Network (CNN)** model trained with **TensorFlow**. The system captures video frames from a camera, processes the images to detect the eyes, and predicts whether the driver is awake or drowsy. If drowsiness is detected, it triggers an alarm and visual indicators.

## Table of Contents

- [Overview](#overview)
- [File Structure](#file-structure)
- [Requirements](#requirements)
- [Setup Instructions](#setup-instructions)
- [Model Training](#model-training)
- [Running the Detection System](#running-the-detection-system)
- [Contributors](#contributors)
- [License](#license)

## Overview

This project aims to monitor the eye movements of drivers in real-time and alert them if they show signs of drowsiness. The main components include:

- **Raspberry Pi** for image capturing and GPIO control
- **OpenCV** for facial and eye detection
- **TensorFlow (Keras)** for machine learning model
- **GPIO pins** for controlling an alarm system (LED, Buzzer)

When the system detects that both eyes are closed for a certain period, it triggers a visual and audio alert to warn the driver.

## File Structure

The project contains the following files and directories:

```
dataset_new/
    |_test/
        |_Closed/
        |_no_yawn/
        |_open/
        |_yawn/
    |_train/
        |_Closed/
        |_no_yawn/
        |_open/
        |_yawn/

haar_cascade_files/
    |_haarcascade_frontalface_alt.xml
    |_haarcascade_lefteye_2splits.xml
    |_haarcascade_righteye_2splits.xml

models/
    |_custmodel.h5

model.ipynb
my_drowsiness_detection.py
```

### Explanation:

- **dataset_new/**: Contains the training and test datasets for the eye detection model.
- **haar_cascade_files/**: Contains the pre-trained Haar cascade files for detecting faces and eyes.
- **models/**: Contains the trained model file `custmodel.h5`.
- **model.ipynb**: Jupyter Notebook where the eye state detection model is trained.
- **my_drowsiness_detection.py**: Python script for real-time drowsiness detection using Raspberry Pi and the trained model.

## Requirements

Before running the project, you need to install the following dependencies:

### Python Libraries

1. **TensorFlow** (for the model)
2. **Keras** (for building and training the CNN model)
3. **OpenCV** (for image processing and eye detection)
4. **NumPy** (for numerical operations)
5. **RPi.GPIO** (for controlling GPIO pins on Raspberry Pi)

You can install the required libraries using `pip`:

```bash
pip install tensorflow keras opencv-python numpy RPi.GPIO
```

Additionally, you will need a **Raspberry Pi** with a camera module and GPIO setup.

## Setup Instructions

1. **Clone the Repository**:

   Clone this repository to your local machine:

   ```bash
   git clone <repository-url>
   cd Driver-Drowsiness-Detection
   ```

2. **Prepare the Dataset**:

   Download or collect images for the "Closed" and "Open" eye states. These images are required for training the model.

3. **Train the Model**:

   Open the `model.ipynb` Jupyter notebook to train the CNN model using the dataset located in the `dataset_new/train/` directory. The model will be saved as `custmodel.h5` in the `models/` directory.

   To train the model:
   - Set the correct path for your dataset in the notebook.
   - Run the notebook cells to preprocess the data, build the model, and train it.

4. **Place Haar Cascade Files**:

   Download the required Haar cascade files for face and eye detection:
   - `haarcascade_frontalface_alt.xml`
   - `haarcascade_lefteye_2splits.xml`
   - `haarcascade_righteye_2splits.xml`

   These files should be placed in the `haar_cascade_files/` directory.

## Model Training

In the `model.ipynb`, the CNN model is created and trained as follows:

1. **Data Preparation**: The dataset of eye images is loaded and preprocessed into a format suitable for training.
2. **Model Architecture**: A simple CNN model with 3 convolutional layers and 2 dense layers is used.
3. **Training**: The model is trained using the **Adam optimizer** and **sparse categorical cross-entropy loss**.

After training, the model is saved as `custmodel.h5` in the `models/` directory.

## Running the Detection System

Once the model is trained, you can run the drowsiness detection system using the `my_drowsiness_detection.py` script on a Raspberry Pi.

1. **Connect the Camera**: Ensure that the Raspberry Pi camera is connected and configured properly.
2. **Run the Detection Script**:

   Execute the following command to start the drowsiness detection:

   ```bash
   python my_drowsiness_detection.py
   ```

   The system will start capturing video frames, detect faces, and predict the eye states. If both eyes are closed for more than a threshold, it will trigger an alarm.

   The GPIO pins are used to control:
   - A **red LED** (BUZ) to indicate a warning.
   - A **green LED** (led) to show active status.
   - A **buzzer** (ALARM_ON) to alert the driver.

   Press `q` to exit the detection loop.


