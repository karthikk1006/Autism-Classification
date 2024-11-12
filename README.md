# Autism Classification Web App

This is a machine learning-based web application designed to predict autism in individuals based on various demographic and screening questionnaire features. The app utilizes different classification models to predict whether an individual is likely to have Autism Spectrum Disorder (ASD). Users can input their data through an interactive sidebar, and the model provides the predicted class (Yes/No) for ASD along with relevant performance metrics and visualizations.

## Features

1. **Screening Questions (A1-A10)**: 
   - The app includes 10 questions based on behavioral attributes like eye contact, social interaction, and play interests, each scored as either 0 or 1.
   
2. **Demographic Information**:
   - Age, gender, ethnicity, and family history of autism.
   - History of jaundice and the individual’s country of residence.
   
3. **Model Selection**:
   - Users can select from various classification models to make predictions.
   - The app displays the model's predicted class and the corresponding accuracy.

4. **Model Evaluation**:
   - Each selected model provides a confusion matrix and a classification report, which are displayed as images on the app.

## Models

The app supports the following classification models:
- Ada Boost
- Decision Tree
- Gaussian Naive Bayes
- K Nearest Neighbours
- Logistic Regression
- MLP (Multi-Layer Perceptron)
- Random Forest
- Ridge Classifier
- SVM (Support Vector Machine)
- XGBoost

## Installation

### Prerequisites

To run the app, you need to install the following dependencies:
- Python 3.x
- Streamlit
- Joblib
- NumPy
- Pandas
- Matplotlib

## Installation

1. **Clone the repository** (if applicable) or download the code.
    ```bash
    git clone <repository-link>
    cd Autism-Classification
    ```

2. **Install dependencies**:
    Ensure you have Python 3.8 or higher installed, and then run:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Streamlit app**:
    ```bash
    streamlit run main.py
    ```

2. **View Predictions**:
    You can enter the details in the models tab and view the analysis and prediction of each model. Additionally, the information regarding the dataset and its analysis are presented in the other two tabs.

## Deployed Link

You can access the app directly at [Autism Classification App](https://autism-classification-prediction.streamlit.app/Models)
