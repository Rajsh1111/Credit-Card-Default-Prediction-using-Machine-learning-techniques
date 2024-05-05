# Credit-Card-Default-Prediction-using-Machine-learning-techniques

Business Context - Banks are primarily known for money lending business. The more money they lend to people whom they can get good interest with timely repayment, the more revenue is for the banks. This not only save banks’ money from having bad loans but also improves image in the public figure and among the regulatory bodies.
The better the banks can identify people who are likely to miss their repayment charges, the more in advance they can take purposeful actions whether to remind them in person or take some strict action to avoid delinquency.
In cases where a borrower is not paying monthly charges when credit is issued against some monetary thing, two terms are frequently used which are delinquent and default.
Delinquent in general is a slightly mild term where a borrower is not repaying charges and is behind by certain months whereas Default is a term where a borrower has not been able to pay charges and is behind for a long period of months and is unlikely to repay the charges.
•	This case study is about identifying the borrowers who are likely to default in the next two years with serious delinquency of having delinquent more than 3 months.
•	We have a general profile about the borrower such as age, Monthly Income, Dependents and the historical data such as what is the Debt Ratio, what ratio of amount is owed with respect to credit limit, and the no of times defaulted in the past one, two, three months.
•	 We will be using all these features to predict whether the borrower is likely to default in the next 2 years or not having delinquency of more than 3 months.
•	These kinds of predictions will help banks to take necessary actions.

•	Objective : Building a model using the inputs/attributes which are general profile and historical records of a borrower to predict whether one is likely to have serious delinquency in the next 2 years
We will be using Python as a tool to perform all kind of operations.
Main Libraries used – 
•	Pandas for data manipulation, aggregation
•	Matplotlib and Seaborn visualization and behavior with respect to the target variable
•	NumPy for computationally efficient operations.
•	Scikit Learn for model training, model optimization and metrics calculation.
•	Imblearn for tackling class imbalance problem.
•	Shap and LIME for model interpretability
•	Keras for Neural Network(Deep Learning architecture)


Project flow chart:

•	Understanding the problem statement
•	Understand the dataset and the behavior of the features/attributes.
•	Performing Exploratory Data Analysis to understand how the data is distributed and what is the behavior of the inputs with respect to target variable which is SeriousDelinquencyin2Years.
•	Data preprocessing will be one based on how the values are distributed such as are there any data entry errors that needed to be removed, outlier treatment, which is necessary for certain algorithms, imputing missing values if there are any.
•	Splitting dataset into the train and test dataset using Stratified Sampling to maintain the event rate across the different datasets so that a model can learn behavior from the training dataset and can predict with certain accuracy up to some on the unseen dataset.
•	Feature Engineering for better decision making by a model.
•	Scaling of the features using BoxCox transformation and Standardization.
•	Training a model using Neural Network as a Deep Learning architecture and analyzing the impact of training on same dataset yet having different features input values because of scaling features, increasing and decreasing minority class.
•	Training a model using statistical technique Logistic Regression and analyzing why scaling features is necessary in such statistical techniques.
•	Training a model using Tree based algorithms such as Bagging and Boosting and analyzing why certain techniques are not required for such algorithms which are quintessential in other modeling techniques. 
•	Hyperparameter tuning of the modeling algorithms and checking its impact on model performance
•	Using Recursive Feature Elimination using Cross Validation to check whether any highly correlated features are there in the model and what are the optimal no of features to be used for training.
•	Analyzing why a popular metric Accuracy will not be useful in our case
•	Checking the model performance on the unseen dataset using metrics such as F1 score, Precision, Recall and the AUC-ROC metric
•	Model Interpretability using SHAP at a global level and LIME at a local level.


### Prerequisites
install python version having 3.7.x as the format else logger syntax might throw an error.

For any package installation use - pip install "package_name"


### Data Overview
The dataset is having 150000 records or 150000 borrowers we are having in the training dataset. We will split the 80% of the data into training and the remaining 20% of the data into test.

The target variable is a binary one having 1 falling in the serious delinquency and 0 for not. There are a mix of continuous and numerical features. Missing values in the data are only in the 2 columns, one having 20% missing values (numerical feature: Monthly Income) and the another one having 3% missing values (categorical feature: Number of Dependents).

6.7% of the customers from the entire dataset are falling in the serious delinquency category which is a classical case of Class Imbalance Problem.


## Code Overview

1. To perform modeling on the dataset, 80% of the data is set as  training data and to check how the model is performing 20% of the data is used as  test data.
2. After doing analysis and visualization of the independent variables/features, there are few major noticeable outliers which need to be removed to avoid fitting issues.

    2.1 Features having the count of behind due date in the emi repayment have a common pattern of outliers which are clipped.
    
    2.2 Removing the rows where data entry errors are noticeable in case of Debt Ratio and removal of rows which do not make any sense as per the description of certain features is carried out as given in step 3 below.

3. Missing value is imputed using the mean and mode as per the type of the feature.

4. Features are engineered to make a model more robust and more informative in decision making to better classify non delinquent borrowers to delinquent borrowers. For that interaction of the features are performed and binary indicators to capture yes/no behaviour.

5. To tackle class imbalance issue upsampling and synthetic samping of the minority class along with the downsampling of the majority class is done to better train a model.

6. Feature Scaling is done so that certain models such as Neural Network and Logistic Regression which rely on features weights can minimise error.

7. For tree based models, no feature scaling is done as performance with or without feature scaling is nearly same.

8. Training of the models using NN, LR, Tree Based Models is done with few techniques to compare results which are feature scaling, tackling class imbalance issue with synthetic sampling and assigning class weights itself in the model by using their hyperparameter.

9. For tree based models, hyperparameters were tuned using GridSearchCV to  have the optimal solution.

10. After doing all the kinds of modeling, it was found that Logistic Regression and NN performs when scaling of the features are done in case scale or range of features are very different from each other.  

11. Tree based models were performing the best and among them LightGBM performance was better after looking at the feature importance and other metrics. Hence, LightGBM was chosen to train the model and predict the unseen data.

12. Metrics were chosen apart from Accuracy to accurately check how the model was performing. 

13. How the model was behaving internally at a global level was checked using SHAP and at a local level was checked using LIME.

## Running the Script

All the indepent operations such as pre processing, training the model are kept in the folder name ML_pipeline and a main script name "engine.py" is used to run the entire code by calling other python scripts from the ML_pipeline folder.

syntax to run - "path_name"/src/python engine.py

Output of the script will be exported to the output folder.
