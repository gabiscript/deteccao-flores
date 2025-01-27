
# Flower Recognition Model

This project involves creating a deep learning model to classify images of flowers into 5 categories: Daisy, Dandelion, Rose, Sunflower, and Tulip. It uses TensorFlow and Keras to build, train, and evaluate a convolutional neural network (CNN) model. The model can predict the type of flower in an image with high accuracy.

## Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Model Usage](#model-usage)
- [License](#license)

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/flower-recognition.git
    cd flower-recognition
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    ```

3. Activate the virtual environment:
    - On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The dataset consists of 5 flower categories:
1. Daisy
2. Dandelion
3. Rose
4. Sunflower
5. Tulip

Images are organized in subfolders corresponding to each flower class. The model is trained using 80% of the dataset and evaluated on the remaining 20%.

## Model Architecture

The model is a Convolutional Neural Network (CNN) with the following layers:
- Data Augmentation (Random Flip, Rotation, Zoom)
- Convolutional layers with ReLU activation functions
- Max-Pooling layers
- Dropout for regularization
- Dense layers for classification

The final layer uses softmax activation to predict one of the 5 flower classes.

## Training

To train the model, you can run the following script:

```bash
python train_model.py
```

This script:
1. Loads and preprocesses the flower images.
2. Performs data augmentation to improve the model's generalization.
3. Defines the model architecture.
4. Compiles the model with the Adam optimizer and SparseCategoricalCrossentropy loss function.
5. Trains the model for 15 epochs.

## Model Usage

Once the model is trained, you can use it to classify new flower images.

1. Load the trained model:
    ```python
    model = tf.keras.models.load_model('Flower_Recog_Model.h5')
    ```

2. Classify an image:
    ```python
    outcome = classify_images('path_to_image')
    print(outcome)
    ```

This will output the predicted flower type along with the confidence score.
