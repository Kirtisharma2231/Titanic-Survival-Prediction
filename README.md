Titanic Survival Prediction (Machine Learning Project)

Project Overview :- 

This project predicts whether a passenger survived the Titanic disaster using machine learning models. The goal is to analyze passenger data and build models that can accurately classify survival outcomes.

>>Dataset:- 

Source: Kaggle Titanic Dataset

>>Features include:-

Age, Sex, Passenger Class, Fare, Embarked, Family Size, Alone

>>Problem Type:-
>>
Binary Classification :- Target Variable: Survived (0 = No, 1 = Yes)

>>Steps Performed:-

1. Data Preprocessing
Handled missing values:
Age → filled with median
Embarked → filled with mode
Cabin → dropped due to excessive missing values
Removed irrelevant columns:
Name, Ticket, PassengerId

2. Feature Engineering
Created new features:
FamilySize = SibSp + Parch + 1
Alone feature derived from FamilySize
Applied one-hot encoding on categorical variables

3. Model Building
The following models were implemented:
1) Logistic Regression
2) Decision Tree Classifier
3) Bagging Classifier (Decision Trees)
4) Random Forest Classifier
   
>> Model Performance

| Model                                  | Test Accuracy| Train Accuracy| Precision| Recall | Remarks              |
|----------------------------------------|--------------|---------------|----------|--------|----------------------|
| Logistic Regression                    | 79.89%       | 80.33%        | 0.82     | 0.85   | Very stable model    |
| Decision Tree                          | 79.89%       | 83.98%        | 0.78     | 0.91   | Slightly overfitted  |
| Bagging (Decision Tree Classifier)     | 80.45%       | 86.51%        | 0.80     | 0.90   | Improved stability   |
| Random Forest                          | 81.01%       | 85.81%        | 0.80     | 0.90   | Best overall model   |

>> Final Model:-

Random Forest was selected as the final model due to:
1) Highest test accuracy (81.01%)
2) Balanced precision and recall
3) Controlled overfitting compared to deeper trees

>> Key Observations:-

1) Logistic Regression showed strong baseline performance with high stability
2) Decision Tree had high recall but lower precision (over-predicting survival)
3) Ensemble methods improved performance by reducing variance

>> Future Improvements:-

1) Feature engineering:
2) Extract titles from names (Mr, Miss, etc.)
3) Age and Fare binning
4) Hyperparameter tuning using GridSearchCV
4) Trying advanced models like XGBoost

>> Technologies Used:-

Python
Pandas, NumPy
Scikit-learn
Matplotlib, Seaborn

>> Conclusion:-

This project demonstrates how different machine learning models perform on a classification problem and highlights the importance of feature engineering and model evaluation.
