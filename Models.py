import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Required Dictionaries
agedescdict={
    "12-15 years" : 0,
    "12-16 years" : 1,
    "18 and more" : 2,
    "4-11 years" : 3
}
ethnicitydict ={
    "Asian":0,
    "Black":1,
    "Hispanic":2,
    "Latino":3,
    "Middle Eastern ":4,
    "Others":5,
    "Pasifika":6,
    "South Asian":7,
    "Turkish":8,
    "White-European":9

}

classdict = {
    0 : "NO",
    1 : "YES"
}

# Model Accuracies
accuracies = {
    "Ada Boost": 0.6405,
    "Decision Tree": 0.5694,
    "Gaussian Naive-Bayes": 0.6348,
    "K Nearest Neighbours": 0.6236,
    "Logistic Regression": 0.6400,
    "MLP": 0.6142,
    "Random Forest": 0.6267,
    "Ridge Classifier": 0.6398,
    "SVM": 0.6398,
    "XGBoost": 0.6485
}

# metrics = {
#     "Ada Boost": {"precision": 0.65, "recall": 0.70, "f1": 0.67},
#     "Decision Tree": {"precision": 0.60, "recall": 0.55, "f1": 0.57},
#     "Gaussian Naive-Bayes": {"precision": 0.62, "recall": 0.65, "f1": 0.63},
#     "K Nearest Neighbours": {"precision": 0.59, "recall": 0.60, "f1": 0.59},
#     "Logistic Regression": {"precision": 0.64, "recall": 0.68, "f1": 0.66},
#     "MLP": {"precision": 0.61, "recall": 0.60, "f1": 0.60},
#     "Random Forest": {"precision": 0.63, "recall": 0.64, "f1": 0.63},
#     "Ridge Classifier": {"precision": 0.61, "recall": 0.62, "f1": 0.61},
#     "SVM": {"precision": 0.65, "recall": 0.67, "f1": 0.66},
#     "XGBoost": {"precision": 0.68, "recall": 0.70, "f1": 0.69}
# }


st.sidebar.title("Model Details")
st.sidebar.subheader("Enter data")
# A1-A10 Screening Question Scores (Binary 0 or 1)
st.sidebar.subheader("Screening Questions Scores (A1-A10)")
A1 = st.sidebar.radio("A1: Does the individual make eye contact when called?", [0, 1])
A2 = st.sidebar.radio("A2: Ability to make eye contact quickly?", [0, 1])
A3 = st.sidebar.radio("A3: Ability to focus on others' conversations?", [0, 1])
A4 = st.sidebar.radio("A4: Interest in sharing activities?", [0, 1])
A5 = st.sidebar.radio("A5: Pretends to care for others?", [0, 1])
A6 = st.sidebar.radio("A6: Interest in social search activities?", [0, 1])
A7 = st.sidebar.radio("A7: Can console others?", [0, 1])
A8 = st.sidebar.radio("A8: Ability to play fantasy games?", [0, 1])
A9 = st.sidebar.radio("A9: Uses cues like 'goodbye'?", [0, 1])
A10 = st.sidebar.radio("A10: Attention to unnecessary details?", [0, 1])

# Additional Dataset Columns
st.sidebar.subheader("Demographic Information")

# Age (0-100)
age = st.sidebar.slider("Age", min_value=0, max_value=100, value=25)

# Gender (M or F)
gender = st.sidebar.radio("Gender", ["m", "f"])

# Ethnicity
ethnicity = st.sidebar.selectbox(
    "Ethnicity", 
    ['Others', 'Middle Eastern', 'White-European', 'Black', 'South Asian', 'Asian', 
     'Pasifika', 'Hispanic', 'Turkish', 'Latino']
)
ethnicity = ethnicitydict[ethnicity]
# Jaundice
jaundice = st.sidebar.radio("History of Jaundice", ["yes", "no"])

# Autism family history
autism = st.sidebar.radio("Family history of autism", ["yes", "no"])

# Country of Residence
country_of_res = st.sidebar.selectbox(
    "Country of Residence", 
    ['Jordan', 'United States', 'Egypt', 'United Kingdom', 'Bahrain', 'Austria', 'Kuwait', 
     'United Arab Emirates', 'Europe', 'Malta', 'Bulgaria', 'South Africa', 'India', 
     'Afghanistan', 'Georgia', 'New Zealand', 'Syria', 'Iraq', 'Australia', 'Saudi Arabia', 
     'Armenia', 'Turkey', 'Pakistan', 'Canada', 'Oman', 'Brazil', 'South Korea', 'Costa Rica', 
     'Sweden', 'Philippines', 'Malaysia', 'Argentina', 'Japan', 'Bangladesh', 'Qatar', 'Ireland', 
     'Romania', 'Netherlands', 'Lebanon', 'Germany', 'Latvia', 'Russia', 'Italy', 'China', 
     'Nigeria', 'U.S. Outlying Islands', 'Nepal', 'Mexico', 'Isle of Man', 'Libya', 'Ghana', 
     'Bhutan', 'American Samoa', 'Albania', 'Belgium', 'Azerbaijan', 'Croatia', 'France', 
     'Indonesia', 'Greenland', 'Bahamas', 'Viet Nam', 'Comoros', 'Portugal', 'Finland', 
     'Norway', 'Anguilla', 'Spain', 'Burundi', 'Chile', 'Tonga', 'Sri Lanka', 'Sierra Leone', 
     'Ethiopia', 'Iran', 'Iceland', 'Nicaragua', 'Hong Kong', 'Ukraine', 'Kazakhstan', 
     'Uruguay', 'Serbia', 'Ecuador', 'Niger', 'Bolivia', 'Aruba', 'Angola', 'Czech Republic', 
     'Cyprus']
)

# Used App Before
used_app_before = st.sidebar.radio("Used App Before", ["yes", "no"])

# Screening Result Score (0-10)
result = st.sidebar.slider("Screening Result Score", min_value=0, max_value=10, value=5)

# Age Description
age_desc = st.sidebar.selectbox("Age Description", ['4-11 years', '12-16 years', '12-15 years', '18 and more'])
age_desc = agedescdict[age_desc]

# Relation to the individual
relation = st.sidebar.selectbox(
    "Relation to the Individual", 
    ['Parent', 'Self', 'Relative', 'Health care professional', 'Others']
)

# Target Variable: Class/ASD
st.sidebar.subheader("Target Variable")
class_asd = st.sidebar.radio("Class/ASD", ["YES", "NO"])


st.sidebar.subheader("Select Model")
option = st.sidebar.selectbox(
    "Choose Model to view",
    ("Ada Boost", "Decision Tree", "Gaussian Naive-Bayes","K Nearest Neighbours","Logistic Regression","MLP","Random Forest","Ridge Classifier","SVM","XGBoost"),
    index = 0,
    placeholder="Select model...",
)
make_prediction_button = st.sidebar.button("Make Prediction")

input_features = np.array([[A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, age, gender, ethnicity, jaundice, autism, result, age_desc, relation]])
column_names = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'age', 'gender','ethnicity', 'jundice', 'austim', 'result', 'age_desc', 'relation']


final_input_df = pd.DataFrame(input_features, columns=column_names)

if make_prediction_button:
    st.title(option)
    
    
    # Run the model based on the selected option
    
    if option == "Ada Boost":
        model = joblib.load("./models/ada_boost.sav")
        prediction = model.predict(final_input_df)
        
        st.markdown(f"<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;'>"
                    f"<b>Predicted Class:</b> {classdict[prediction[0]]}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;'>"
                    f"<b>Model Accuracy:</b> {accuracies['Ada Boost']}</div>", unsafe_allow_html=True)
        
        # Display Confusion Matrix Image
        st.image("./Confusion_Matrix/AdaBoost.png", caption="Confusion Matrix: Ada Boost", use_container_width=True)
    
        # Display Classification Report Image
        st.image("./classification_reports/AdaBoost.png", caption="Classification Report: Ada Boost", use_container_width=True)
    
    
    elif option == "Decision Tree":
        model = joblib.load("./models/decision_tree.sav")
        prediction = model.predict(final_input_df)
        
        st.markdown(f"<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;'>"
                    f"<b>Predicted Class:</b> {classdict[prediction[0]]}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;'>"
                    f"<b>Model Accuracy:</b> {accuracies['Decision Tree']}</div>", unsafe_allow_html=True)
        
        # Display Confusion Matrix Image
        st.image("./Confusion_Matrix/DecisionTrees.png", caption="Confusion Matrix: Decision Tree", use_container_width=True)
    
        # Display Classification Report Image
        st.image("./classification_reports/DecisionTrees.png", caption="Classification Report: Decision Tree", use_container_width=True)
    
    
    elif option == "Gaussian Naive-Bayes":
        model = joblib.load("./models/gaussian_nb.sav")
        prediction = model.predict(final_input_df)
        
        st.markdown(f"<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;'>"
                    f"<b>Predicted Class:</b> {classdict[prediction[0]]}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;'>"
                    f"<b>Model Accuracy:</b> {accuracies['Gaussian Naive-Bayes']}</div>", unsafe_allow_html=True)
        
        # Display Confusion Matrix Image
        st.image("./Confusion_Matrix/GaussianNB.png", caption="Confusion Matrix: Gaussian Naive-Bayes", use_container_width=True)
    
        # Display Classification Report Image
        st.image("./classification_reports/GaussianNB.png", caption="Classification Report: Gaussian Naive-Bayes", use_container_width=True)
    
    
    elif option == "K Nearest Neighbours":
        model = joblib.load("./models/knn.sav")
        prediction = model.predict(final_input_df)
        
        st.markdown(f"<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;'>"
                    f"<b>Predicted Class:</b> {classdict[prediction[0]]}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;'>"
                    f"<b>Model Accuracy:</b> {accuracies['K Nearest Neighbours']}</div>", unsafe_allow_html=True)
        
        # Display Confusion Matrix Image
        st.image("./Confusion_Matrix/KNN.png", caption="Confusion Matrix: K Nearest Neighbours", use_container_width=True)
    
        # Display Classification Report Image
        st.image("./classification_reports/KNN.png", caption="Classification Report: K Nearest Neighbours", use_container_width=True)
    
    
    elif option == "Logistic Regression":
        model = joblib.load("./models/logistic_regression.sav")
        prediction = model.predict(final_input_df)
        
        st.markdown(f"<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;'>"
                    f"<b>Predicted Class:</b> {classdict[prediction[0]]}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;'>"
                    f"<b>Model Accuracy:</b> {accuracies['Logistic Regression']}</div>", unsafe_allow_html=True)
        
        # Display Confusion Matrix Image
        st.image("./Confusion_Matrix/Logistic Regression.png", caption="Confusion Matrix: Logistic Regression", use_container_width=True)
    
        # Display Classification Report Image
        st.image("./classification_reports/LogisticRegression.png", caption="Classification Report: Logistic Regression", use_container_width=True)
    
    
    elif option == "MLP":
        model = joblib.load("./models/mlp.sav")
        prediction = model.predict(final_input_df)
        
        st.markdown(f"<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;'>"
                    f"<b>Predicted Class:</b> {classdict[prediction[0]]}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;'>"
                    f"<b>Model Accuracy:</b> {accuracies['MLP']}</div>", unsafe_allow_html=True)
        
        # Display Confusion Matrix Image
        st.image("./Confusion_Matrix/MLP.png", caption="Confusion Matrix: MLP", use_container_width=True)
    
        # Display Classification Report Image
        st.image("./classification_reports/MLP.png", caption="Classification Report: MLP", use_container_width=True)
    
    elif option == "Random Forest":
        model = joblib.load("./models/random_forest.sav")
        prediction = model.predict(final_input_df)
        
        st.markdown(f"<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;'>"
                    f"<b>Predicted Class:</b> {classdict[prediction[0]]}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;'>"
                    f"<b>Model Accuracy:</b> {accuracies['Random Forest']}</div>", unsafe_allow_html=True)
        
        # Display Confusion Matrix Image
        st.image("./Confusion_Matrix/RandomForest.png", caption="Confusion Matrix: Random Forest", use_container_width=True)
    
        # Display Classification Report Image
        st.image("./classification_reports/RandomForest.png", caption="Classification Report: Random Forest", use_container_width=True)
    
    
    elif option == "Ridge Classifier":
        model = joblib.load("./models/ridge_classifier.sav")
        prediction = model.predict(final_input_df)
        
        st.markdown(f"<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;'>"
                    f"<b>Predicted Class:</b> {classdict[prediction[0]]}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;'>"
                    f"<b>Model Accuracy:</b> {accuracies['Ridge Classifier']}</div>", unsafe_allow_html=True)
        
        # Display Confusion Matrix Image
        st.image("./Confusion_Matrix/RidgeClassifier.png", caption="Confusion Matrix: Ridge Classifier", use_container_width=True)
    
        # Display Classification Report Image
        st.image("./classification_reports/RidgeClassifier.png", caption="Classification Report: Ridge Classifier", use_container_width=True)
    
    
    elif option == "SVM":
        model = joblib.load("./models/svc.sav")
        prediction = model.predict(final_input_df)
        
        st.markdown(f"<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;'>"
                    f"<b>Predicted Class:</b> {classdict[prediction[0]]}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;'>"
                    f"<b>Model Accuracy:</b> {accuracies['SVM']}</div>", unsafe_allow_html=True)
        
        # Display Confusion Matrix Image
        st.image("./Confusion_Matrix/SVC.png", caption="Confusion Matrix: SVM", use_container_width=True)
    
        # Display Classification Report Image
        st.image("./classification_reports/SVC.png", caption="Classification Report: SVM", use_container_width=True)
    
    elif option == "XGBoost":
        model = joblib.load("./models/xgb.sav")
        prediction = model.predict(final_input_df)
        
        st.markdown(f"<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;'>"
                    f"<b>Predicted Class:</b> {classdict[prediction[0]]}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;'>"
                    f"<b>Model Accuracy:</b> {accuracies['XGBoost']}</div>", unsafe_allow_html=True)
        
        # Display Confusion Matrix Image
        st.image("./Confusion_Matrix/XGB.png", caption="Confusion Matrix: XGBoost", use_container_width=True)
    
        # Display Classification Report Image
        st.image("./classification_reports/XGB.png", caption="Classification Report: XGBoost", use_container_width=True)
    