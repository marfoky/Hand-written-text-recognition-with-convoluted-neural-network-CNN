# Handwritten Text Recognition Using Convolutional Neural Networks (CNN)
Abstract
This project focuses on building a Convolutional Neural Network (CNN) for handwritten text recognition using the MNIST dataset. The goal is to classify handwritten digits (0-9) from 28x28 pixel grayscale images. The CNN model employs convolutional, pooling, and dense layers, combined with dropout regularization and data augmentation techniques, to achieve high performance in digit recognition. The project discusses the methodology, model architecture, training process, and results, demonstrating the model's accuracy and potential areas for improvement.

Methodology
Introduction
Handwritten text recognition plays a crucial role in various applications like digitizing handwritten documents and automated data entry. Using the MNIST dataset of 60,000 training images and 10,000 test images, this project aims to develop a CNN model capable of recognizing handwritten digits. Advanced data augmentation and regularization techniques are employed to ensure high accuracy and reliability in real-world scenarios.

Technology Used
Languages: Python
Libraries: TensorFlow, Keras, Pandas, NumPy, Matplotlib
Environment: Jupyter Notebook
Data Preparation
The MNIST dataset was normalized to the range [0, 1] and reshaped for single-channel (grayscale) images. The dataset was split into training (60,000 images) and test sets (10,000 images), and initial visualizations were plotted for better understanding.

Data Augmentation
Keras' ImageDataGenerator was used for augmenting the training data by applying transformations such as rotation, shifting, and zooming to improve the model's generalization.

Model Architecture
The CNN model includes:

Input Layer: 28x28 pixel grayscale images
Convolutional Layers: 3 layers with filters (32, 64, 64) and 3x3 kernels using ReLU activation
MaxPooling Layers: 3 layers with 2x2 pool size
Flatten Layer: Converts the 3D output of the final pooling layer to 1D
Dense Layers: Two fully connected layers (64 neurons each), followed by a 50% dropout layer
Output Layer: Softmax layer with 10 neurons for digit classification
Model Compilation and Training
The model was compiled using the Stochastic Gradient Descent (SGD) optimizer with a learning rate of 0.01, momentum of 0.9, and sparse categorical cross-entropy as the loss function. The model was trained for 50 epochs with a batch size of 32, employing early stopping to prevent overfitting.

Regularization Method
Dropout regularization was used with a rate of 0.5 in dense layers to prevent overfitting by randomly ignoring 50% of the neurons during each training iteration.

Results
The CNN model achieved the following performance metrics:

Metric	Training Set	Validation Set	Test Set
Accuracy	99.87%	99.32%	99.56%
Loss	0.0039	0.0564	0.0158
The model demonstrated high accuracy and low loss across all datasets, indicating its effectiveness in recognizing handwritten digits with minimal overfitting.

Conclusion
The CNN model developed in this project successfully recognized handwritten digits with high accuracy. The architecture, combined with data augmentation and regularization, resulted in a robust and generalizable model. These findings underscore the power of CNNs for image classification tasks and set the stage for future improvements.

Future Work
Future enhancements could include:

Exploring more advanced CNN architectures like ResNet or DenseNet
Applying additional data augmentation techniques
Investigating alternative regularization methods (e.g., L2 regularization, batch normalization)
Testing the model on more diverse datasets
Hyperparameter tuning using grid search or random search