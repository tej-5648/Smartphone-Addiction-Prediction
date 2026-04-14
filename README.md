# Smartphone Addiction Prediction
**CS304N/354N Computational Intelligence - Course Project**

## Group Members
* **Md Asif Hussain** (230041021)
* **N Sai Sathwik** (230041024)
* **P Sai Prakul** (230041031)
* **K Vivek Tej** (230041014)
* **V Akshay** (230041039)

---

## Project Overview
This project presents a comparative study of Machine Learning (ML) and Computational Intelligence (CI) techniques for predicting smartphone addiction based on user behavioral and demographic data. By exploring both unsupervised and supervised models, we evaluate the best approach to detect addiction patterns.

## Dataset Details
The model is trained on `Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv`.
* **Size:** 7500 Rows, 16 Columns.
* **Features:** Demographics (Age, Gender), Screen time metrics (Daily hours, Social media, Gaming, Weekend screen time), Health metrics (Sleep hours, Stress level), and App engagement (Notifications, App opens).
* **Target Variable:** `addicted_label` (Binary: 0 for Not Addicted, 1 for Addicted). 

## Project Pipeline
1. **Exploratory Data Analysis (EDA):** Analyzed class balance (ratio of 0.41), feature distributions, and correlations using `matplotlib` and `seaborn`.
2. **Preprocessing & Feature Engineering:** * Label Encoding for categorical variables.
   * Standard Scaling (`StandardScaler`) for numeric features.
   * Stratified 80/20 train-test split to maintain class balance.
3. **Unsupervised Learning:** Applied **K-Means Clustering** to test if addiction structure could be recovered without labels.
4. **Supervised Learning:** Trained and evaluated multiple models:
   * *Linear:* Perceptron, Ridge Regression, Logistic Regression.
   * *Non-linear:* SVM (Linear, RBF kernels), Decision Tree, KNN.
   * *Ensemble:* Random Forest, Gradient Boosting, AdaBoost.
5. **Evaluation & Cross-Validation:** Models were evaluated using Accuracy, Precision, Recall, F1-Score, and ROC-AUC. Stability was verified using 5-fold Stratified Cross-Validation.

---

## Setup and Installation (How to Run)

### Prerequisites
Make sure you have Python 3.8+ installed. You will need the following libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```
### Execution Steps
1. Clone or download the project repository to your local machine.
2. Ensure the dataset `Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv` is placed in the **same directory** as the Jupyter Notebook.
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook "smartphone_addiction_project (2).ipynb"
   ```
4.Click on Kernel > Restart & Run All to execute the pipeline from start to finish.

## Verification & Expected Outputs
As the notebook runs, you can verify the execution through the following outputs:
* **Cell 5 (EDA):** Look for generated plots including `fig_target_distribution.png` showing the target class distribution.
* **Cell Preprocessing:** The output will print `Train size: 6000` and `Test size: 1500`.
* **Model Training Cells:** Terminal outputs will display classification reports (Precision, Recall, F1-Score) and confusion matrices for each algorithm.
* **Cross-Validation Cell:** At the end of the notebook, a comparative table and a bar plot will be generated, displaying the Mean CV F1-Score ± Standard Deviation for all supervised models.

---

## Key Insights & Conclusions
* **Best Performing Models:** **Gradient Boosting** and **SVM (RBF kernel)** proved to be the most robust, showing stable and consistent performance (lowest variance) across cross-validation folds. The RBF kernel successfully captured complex non-linear behavioral relationships.
* **Feature Importance:** Ridge Regression highlighted that **usage intensity features** (total screen time, frequency of app opens) are the strongest predictors of addiction. 
* **Poor Performers:** **Perceptron** performed the worst due to its strictly linear nature. Unsupervised **K-Means** struggled to capture complex decision boundaries without label information.
