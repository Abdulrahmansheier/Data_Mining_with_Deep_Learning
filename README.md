# Data_Mining_with_Deep_Learning
Detect Covid-19 from Chest X-Ray Images using neural networks.


In this project, the primary task was to detect COVID-19 from chest X-ray images using neural networks. The work was divided into two subtasks:

**Subtask 1:**
The first subtask involved developing deep learning models to classify COVID-19 and normal chest X-ray images. The following steps were undertaken:
1. Data Exploration: The dataset was explored, and six random images from the training folder were plotted.
2. Data Preprocessing: Image preprocessing involved resizing images, converting pixel values to float, normalizing the images, and converting data into NumPy arrays or tensors. Data augmentation was also applied, including resizing, rescaling, random flipping, and rotation.
3. Label Encoding: Text labels were converted into numeric codes (COVID-19 as 0 and normal as 1).
4. Convolutional Neural Network (CNN) Model: A CNN model was implemented with specific hyperparameters, including layers with 32, 64, and 128 filters, ReLU activation, dropout layers, flattening, and a dense output layer with sigmoid activation. Model checkpointing, early stopping, weight decay, and learning rate optimization were used.
5. Model Investigation: The CNN model's performance was analyzed, and adjustments were made to overcome overfitting issues. Learning rate adjustments and early stopping were found to be effective.

A Feedforward Neural Network (FNN) model was also implemented, but it did not perform as well as the CNN model. FNN's accuracy remained consistent at a lower level (around 50%), and various techniques such as data augmentation, weight decay, early stopping, ensembles, and dropout did not significantly impact its performance.

**Subtask 2:**
In the second subtask, transfer learning with the ResNet-50 model was applied. The pre-trained weights of ResNet-50 on the ImageNet dataset were used. The following steps were taken:
1. Transfer Learning Setup: The pre-trained convolutional base of ResNet-50 was added as a layer to a new model. The pre-trained weights were frozen to prevent updating during training.
2. Model Adaptation: The last few layers of the pre-trained model were removed, and two new layers were added to adapt the model to the new dataset.
3. Dropout Experiment: It was observed that removing the dropout function helped prevent overfitting.

In summary, the CNN model performed better than the FNN model for image classification, especially in the context of chest X-ray images. CNNs are more suitable for image classification because they can efficiently identify local patterns and spatial relationships within images, while FNNs treat images as flattened vectors of pixel values and struggle to extract spatial information. Transfer learning with the ResNet-50 model was also explored, which allowed the model to leverage pre-trained features from ImageNet for improved performance.
