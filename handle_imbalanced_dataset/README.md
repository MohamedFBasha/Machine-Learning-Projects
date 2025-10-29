⚖️ Handling Imbalanced Datasets in Machine Learning
🧠 Overview

This project demonstrates various techniques to handle imbalanced datasets — a common challenge in machine learning classification tasks where one class significantly outweighs others.
The notebook explores multiple resampling methods, ensemble techniques, and advanced hybrid SMOTE variants to improve model performance and ensure fair learning.

📋 Table of Contents

Cross Validation (K-Fold)

Random Forest Classifier

Random Forest Classifier with class_weight

Under Sampling

Over Sampling

SMOTE + Tomek Links

Easy Ensemble

Adaptive SMOTE

SMOTE + ENN (Edited Nearest Neighbors)

SMOTE + Borderline

KMeans SMOTE

SVM SMOTE

Anomaly Detection

⚙️ Project Description
🎯 Objective

To compare and evaluate different balancing techniques and determine which method achieves the best trade-off between recall, precision, and overall model robustness.

🧩 Models Used

RandomForestClassifier (with and without class_weight)

Additional models tested under resampling strategies

🧪 Validation

Used K-Fold Cross Validation for consistent model evaluation

Metrics evaluated: Accuracy, Precision, Recall, F1-score, ROC-AUC

🧮 Techniques Applied
🔹 1. Under Sampling

Reducing majority class samples to balance the dataset.

🔹 2. Over Sampling

Duplicating or generating synthetic data for the minority class.

🔹 3. SMOTE Variants

SMOTETomek: Combines SMOTE oversampling and Tomek Links undersampling

Adaptive SMOTE: Dynamically adjusts synthetic sample generation

SMOTE + ENN: Cleans noisy samples after oversampling

Borderline-SMOTE: Focuses on hard-to-classify samples near decision boundaries

KMeans SMOTE: Uses clustering to generate better synthetic samples

SVM SMOTE: Uses support vectors to guide sample creation

🔹 4. Ensemble Method

Easy Ensemble: Trains multiple classifiers on balanced subsets to improve minority prediction.

🔹 5. Anomaly Detection

Used as an alternative approach by identifying minority samples as anomalies instead of using traditional balancing.

📊 Evaluation

Each technique was compared based on:

Class distribution before and after balancing

Model performance metrics

ROC-AUC curves and confusion matrices

The results highlight how resampling can significantly enhance minority class recall while maintaining general accuracy.

🧠 Key Insights

Adaptive SMOTE and SMOTE + ENN provided the best overall balance.

Random Forest with class weights performed competitively without resampling.

Anomaly detection was effective in certain highly skewed datasets.

Combining ensemble + resampling achieved the most stable outcomes.

💡 Requirements
pip install scikit-learn imbalanced-learn numpy pandas matplotlib seaborn

🧾 How to Run

Clone this repository

Install dependencies

Open and run the notebook:

jupyter notebook Handling_Imbalanced_Dataset.ipynb


Review evaluation metrics and visualizations.

📈 Visualization

The notebook includes plots for:

Class distribution before/after balancing

Confusion matrices

ROC-AUC curves

Feature importance from Random Forest

🧩 Tech Stack

Python 3.10+

Scikit-learn

Imbalanced-learn (imblearn)

Matplotlib / Seaborn

NumPy / Pandas
