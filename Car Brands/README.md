# Car Brands Classification

## Overview
This project focuses on classifying car brands using deep learning. A transfer learning approach was utilized with the **ReXNet_150** model. The training process included early stopping to prevent overfitting.

## Dataset
The dataset consists of images of various car brands. The images were preprocessed and split into training and validation sets to ensure a balanced evaluation.

## Model
- **Architecture:** ReXNet_150 (Pretrained model for transfer learning)
- **Training Accuracy:** 99%
- **Best Validation Accuracy:** 87.6%
- **Training Duration:** 13 epochs (Early stopping applied)

## Training Details
- **Optimizer:** Adam
- **Loss Function:** Cross-Entropy Loss
- **Batch Size:** Optimized for best performance
- **Learning Rate:** Adjusted dynamically
- **Early Stopping:** Triggered after 13 epochs to prevent overfitting

## Results
The model achieved a high training accuracy of **99%**, while the best validation accuracy recorded was **87.6%**. The early stopping mechanism ensured that training stopped once no significant improvement was observed.

## Usage
To use the model for classification:
1. Load the trained ReXNet_150 model.
2. Preprocess the image (resize, normalize, etc.).
3. Pass the image through the model.
4. Retrieve the predicted class label.

## Conclusion
This project successfully implemented transfer learning for car brand classification. While training accuracy was very high, the validation accuracy suggests some room for improvement, possibly through data augmentation or fine-tuning hyperparameters.

## Future Improvements
- Data augmentation for better generalization
- Experimenting with different learning rates and optimizers
- Testing on a larger dataset for improved accuracy

