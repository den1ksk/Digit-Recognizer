#### English Version | [Kaggle competition](https://www.kaggle.com/competitions/digit-recognizer)

# Digit Recognizer - Kaggle Competition

### Competition Description

MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.

### Goal

In this competition, your goal is to correctly identify digits from a dataset of tens of thousands of handwritten images. We’ve curated a set of tutorial-style kernels which cover everything from regression to neural networks. We encourage you to experiment with different algorithms to learn first-hand what works well and how techniques compare.

### Approach

This project uses deep learning techniques to classify handwritten digits. The following steps were taken:

1. **Data exploration:**
   - The dataset was explored to understand its structure, with a focus on the pixel values of each image (28x28 pixel grid).
   - Visualizations such as sample digit images were used to gain insights into the dataset.

2. **Data Preprocessing:**
   - The pixel values were normalized by scaling them between 0 and 1.
   - The images were reshaped into 28x28 matrices to prepare them for the convolutional neural network (CNN).
   - The target variable was one-hot encoded into 10 categories for each digit (0–9).

3. **Model Architecture:**
   - A convolutional neural network (CNN) was built using layers of convolution, batch normalization, pooling, and dropout for regularization.
   - The model uses a combination of `Conv2D`, `MaxPooling2D`, `Dropout`, and `Dense` layers, and is optimized using the Adam optimizer.
   - `ReduceLROnPlateau` and `EarlyStopping` callbacks were employed to improve training efficiency and prevent overfitting.

4. **Model Training:**
   - The dataset was split into training and validation sets, and the model was trained over 50 epochs with a batch size of 64.
   - Training progress was monitored, and accuracy and loss were tracked for both training and validation sets.

5. **Model Evaluation:**
   - The model's performance was evaluated using metrics such as accuracy and loss. Training and validation accuracy reached approximately 99.5%.
   - Training history was plotted to visualize model accuracy and loss across epochs.

6. **Prediction and Submission:**
   - After training, predictions were made on the test dataset. The model's performance ranked in the top 5% on the Kaggle leaderboard with a score of **0.99553**.
   - The results were saved in a submission file for the competition.

### Results

The model achieved a Kaggle score of **0.99553**, placing it in the **top 5%** of the competition leaderboard.

### Files

- `Digit Recognizer.ipynb`: Jupyter Notebook containing the complete code and analysis.
- `submission.csv`: File containing the predictions for the test dataset.

### Libraries Used

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `tensorflow`
- `scikit-learn`

### Acknowledgements

This project was completed as part of the Kaggle competition. Special thanks to Kaggle for providing the dataset and platform for this challenge.
