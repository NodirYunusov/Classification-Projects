# Rice Disease Classification

This project focuses on classifying rice diseases using deep learning. The model is trained on a dataset of rice leaf images to identify various diseases accurately.

## Model Details
- **Architecture:** ReXNet_150
- **Training Accuracy:** 99%
- **Best Validation Accuracy:** 98%
- **Epochs Trained:** 7 (Training stopped early due to stable validation accuracy)

## Dataset
The dataset consists of labeled images of rice leaves with different disease conditions. The images are preprocessed and augmented to enhance model generalization.

## Training
The model was trained using ReXNet_150 with:
- Data augmentation for better generalization
- Adam optimizer for efficient weight updates
- Categorical cross-entropy loss for multi-class classification
- Early stopping to prevent overfitting

## Results
The model achieved:
- **Training Accuracy:** 99%
- **Best Validation Accuracy:** 98%


