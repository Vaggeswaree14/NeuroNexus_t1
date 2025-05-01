# NeuroNexus_t1
Titanic Survival Prediction 🚢 | Machine Learning Project
This project is a classic example of binary classification using machine learning to predict the survival of passengers aboard the Titanic. It is built as part of the NeuroNexus Task 1, demonstrating the application of supervised learning techniques on the popular Titanic dataset from Kaggle.

📌 Project Overview
The objective of this project is to build a machine learning model that predicts whether a passenger survived the Titanic disaster based on features such as age, gender, ticket class, number of siblings/spouses aboard, and more.

The notebook includes:

Data exploration and visualization

Feature engineering and preprocessing

Training multiple ML models

Evaluation and comparison of model performance

Hyperparameter tuning for improved accuracy

📁 Dataset
The dataset used is the Titanic dataset from Kaggle. It includes the following files:

train.csv: Used to train the machine learning model.

test.csv: Used for testing and generating predictions.

Key features in the dataset:

Pclass: Passenger class (1st, 2nd, 3rd)

Sex: Gender

Age: Age in years

SibSp: Number of siblings/spouses aboard

Parch: Number of parents/children aboard

Fare: Passenger fare

Embarked: Port of Embarkation (C, Q, S)

🛠️ Technologies Used
Python 🐍

Pandas & NumPy for data manipulation

Matplotlib & Seaborn for data visualization

Scikit-learn for machine learning models and evaluation

Jupyter Notebook for interactive development

📊 Models Implemented
Logistic Regression

Decision Tree

Random Forest

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Each model is evaluated using:

Accuracy score

Confusion matrix

Classification report (Precision, Recall, F1-score)

🧪 Evaluation & Results
After training and comparing the models, the best-performing one is selected based on accuracy and F1-score. Hyperparameter tuning is also performed using GridSearchCV to further optimize the selected model.

📈 Sample Visualizations
The notebook includes insightful plots such as:

Survival distribution based on gender and class

Age and fare distribution

Heatmaps showing correlation between features

🧠 Conclusion
This project successfully demonstrates the end-to-end workflow of a supervised classification task. It highlights the importance of data preprocessing, feature selection, and model evaluation in building an effective predictive model.
