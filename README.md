# MNIST Handwritten Digit Classification

![image](https://github.com/SaadARazzaq/MNIST-Classification-Problem/assets/123338307/17d882be-a403-4df7-9319-f41975aee384)

## Introduction

This repository provides an in-depth analysis and code for the MNIST handwritten digit classification problem. The MNIST dataset is a classic benchmark in the field of machine learning, consisting of 28x28 pixel grayscale images of handwritten digits (0-9). The objective is to develop a model capable of recognizing and classifying these digits accurately.

## Approach and Intuition

### Dataset

The MNIST dataset is loaded using the `keras.datasets.mnist.load_data()` function. The dataset is divided into a training set (60,000 samples) and a test set (10,000 samples).

### Data Preprocessing

**Flattening Images:** Each image in the dataset is a 28x28 pixel matrix. To feed them into machine learning models, we flatten these images into 1D vectors of length 784. This simplifies the data while preserving essential information.

**Normalization:** The pixel values in the images are normalized to a range between 0 and 1 by dividing each pixel value by 255. This scaling ensures all input features have the same range and makes training more efficient.

### Linear SVM Classifier

We start with a Linear Support Vector Machine (SVM) classifier, a powerful tool for binary and multiclass classification tasks. The intuition behind using SVM for MNIST classification is to find a hyperplane that best separates the different digit classes in a high-dimensional space.

**Training:** The SVM classifier is trained on the preprocessed training data. It tries to find the optimal hyperplane that maximizes the margin between different classes.

**Cross-Validation:** The model's performance is assessed using cross-validation. This technique involves dividing the training data into subsets for training and testing, which helps estimate how well the model will generalize to unseen data.

### Linear SVM Classifier with Standardization

To improve the SVM classifier's performance, we standardize the data using `StandardScaler`. Standardization transforms the data so that it has a mean of 0 and a standard deviation of 1.

**Standardization:** Standardization is particularly useful for SVMs because it ensures that all features have the same scale. SVMs are sensitive to the scale of input features, and standardization helps achieve a balanced influence of all features on the decision boundary.

**Training Accuracy:** We report the training accuracy of the SVM classifier after standardization.

### Non-Linear SVM Classifier with RBF Kernel

While linear SVMs work well for many problems, some datasets, like MNIST, are better tackled with non-linear models. We explore a Non-Linear SVM classifier using the Radial Basis Function (RBF) kernel.

**RBF Kernel:** The RBF kernel is used to capture complex relationships between data points in a higher-dimensional space. It can model intricate patterns that linear SVMs cannot.

**Training:** The non-linear SVM classifier with the RBF kernel is trained on the MNIST data, potentially providing a better solution for this image classification problem.

### Model Evaluation

The trained classifiers are then evaluated using the test dataset. We assess the model's performance using classification reports, which provide metrics such as precision, recall, and F1-score for each digit class. This offers insights into the model's strengths and weaknesses for different digits.

### Confusion Matrices

Additionally, we visualize the confusion matrices to understand how well the classifiers are performing for each class. Confusion matrices help identify areas where the models may be making errors in classification.

## Usage

To utilize the code in this repository for MNIST digit classification, follow these steps:

1. Clone the repository to your local machine.

2. Open the project in your preferred Python environment, such as Jupyter Notebook or any Python IDE. I recommend using Google Colab.

3. Install the necessary dependencies using the following command:

   ```bash
   pip install numpy pandas scikit-learn matplotlib keras

4. Load the MNIST dataset using the provided code.

5. Run the code cells step by step to execute the different classifiers and evaluate their performance.

6. Analyze the classification reports, confusion matrices, and training accuracies to gain insights into the classifiers' capabilities and shortcomings.

## Conclusion

MNIST digit classification is a fundamental machine learning problem, and it serves as a stepping stone for understanding more complex image classification tasks. The combination of linear and non-linear SVMs allows us to explore different approaches and understand their strengths and limitations in this context.

## Contributing

Contributions to this repository are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## Contact

For any inquiries or questions, you can reach out to the project maintainer:

Name: [Saad Abdur Razzaq]

Email: [sabdurrazzaq124@gmail.com]

Linkedin: [Let's Connect](https://www.linkedin.com/in/saadarazzaq/)

Feel free to get in touch!

```bash
Made with 💖 by Saad
```
