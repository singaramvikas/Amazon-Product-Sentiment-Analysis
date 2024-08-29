# Amazon Sentiment Analysis

This project performs sentiment analysis on Amazon product reviews using machine learning techniques. It includes data preprocessing, feature extraction, model training, and evaluation to classify reviews into positive or negative sentiments and handle imbalanced data.

## Contents

- **Data Preprocessing**: Cleaning and preparing the data for analysis
- **Feature Extraction**: Converting text data into numerical features using CountVectorizer and TF-IDF
- **Model Training**: Training and evaluating various classification models
- **Handling Imbalanced Data**: Applying techniques to balance class distributions
- **Model Optimization**: Using grid search to find the best model parameters

## Requirements

To run this notebook, you will need the following Python libraries:
- `pandas`
- `numpy`
- `seaborn`
- `sklearn`
- `imblearn` (for handling imbalanced data)
- `matplotlib`

You can install these libraries using pip:

```bash
pip install pandas numpy seaborn scikit-learn imbalanced-learn matplotlib
```

## Getting Started

1. **Download the Data**

   Ensure you have the dataset `Reviews.csv`. Update the path in the code to the location where your dataset is stored.

2. **Run the Notebook**

   The notebook is designed to be run in a Jupyter notebook environment, such as Google Colab or Jupyter Notebook. To run the notebook:

   - Open the notebook in Google Colab or Jupyter Notebook.
   - Ensure that all necessary libraries are installed.
   - Execute each cell in the notebook sequentially.

## Data Description

The dataset contains Amazon product reviews with the following columns:
- `Id`: Review identifier
- `Text`: Review text
- `Score`: Rating score (1 to 5)
- `HelpfulnessNumerator`: Number of helpful votes
- `HelpfulnessDenominator`: Total number of votes

## Steps Overview

1. **Data Preprocessing**
   - Calculate the helpful percentage (`Helpful%`) and categorize it into bins.
   - Create a pivot table and visualize it using a heatmap.

2. **Feature Extraction**
   - Use `CountVectorizer` and `TfidfVectorizer` to transform text data into feature matrices.

3. **Model Training and Evaluation**
   - Train a `LogisticRegression` model and evaluate its accuracy.
   - Extract and display the top 20 positive and negative words based on model coefficients.
   - Perform predictions and evaluate using confusion matrices and accuracy scores.

4. **Handling Imbalanced Data**
   - Apply `RandomOverSampler` to balance the dataset.
   - Perform model optimization using grid search with cross-validation.

## Results

- **Confusion Matrix**: Visualizes the performance of the model in predicting the sentiment of reviews.
- **Classification Report**: Provides precision, recall, and F1 scores for the classification model.
- **Top Words**: Lists the top 20 positive and negative words based on the model’s coefficients.
- **Model Accuracy**: Displays the accuracy score of the Logistic Regression model and other models.

## Acknowledgments

- Thanks to the contributors and developers of the Python libraries used in this project.
- The dataset is publicly available and used for sentiment analysis in this notebook.

Feel free to contribute to this project or use it as a reference for your own work. If you have any questions or issues, please open an issue on this repository.
