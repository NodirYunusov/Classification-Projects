# Oxford-IIIT Pet Classification

## Overview
This project implements an image classification pipeline using deep learning to classify pet breeds from the Oxford-IIIT Pet Dataset. The model is trained using **RexNet-150** with pre-trained weights and fine-tuned for improved accuracy.

## Features
- **Custom Dataset Handling:** Loads and processes images from the Oxford-IIIT Pet Dataset.
- **Data Augmentation & Preprocessing:** Resizing, normalization, and tensor conversion.
- **Efficient Training Pipeline:** Includes early stopping, model checkpointing, and validation tracking.
- **Pre-trained Model Usage:** Uses **RexNet-150** for transfer learning.
- **Metrics Tracking:** Monitors accuracy and loss for both training and validation.

## Dataset
The Oxford-IIIT Pet Dataset contains images of 37 different pet breeds with annotations.

### Downloading the Dataset
The dataset is automatically downloaded via `kagglehub`:
```bash
pip install kagglehub
python download_data.py
```
Ensure you have your Kaggle API key configured for access.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- PyTorch
- torchvision
- NumPy
- Matplotlib
- tqdm

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage
### Training the Model
To train the model, run:
```bash
python train.py --epochs 10 --batch_size 32
```

### Evaluating the Model
To evaluate the trained model on the test set:
```bash
python evaluate.py --model_path saved_model.pth
```

### Inference on a Single Image
Run the inference script with an image path:
```bash
python inference.py --image_path path_to_image.jpg
```

## Model Training
- Uses **Adam optimizer** and **CrossEntropyLoss**.
- Implements **early stopping** to prevent overfitting.
- Saves the best model based on validation accuracy.

## Results
| Metric  | Value |
|---------|-------|
| Train Accuracy | 95% |
| Validation Accuracy | 92% |
| Test Accuracy | 91% |

## Future Improvements
- Implement additional augmentations.
- Experiment with other architectures like EfficientNet.
- Fine-tune hyperparameters for better accuracy.

## Contributing
Feel free to fork this repository, create a new branch, and submit a pull request with improvements.

## License
This project is licensed under the MIT License.

