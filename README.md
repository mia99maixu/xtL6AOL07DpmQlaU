# Term Deposit Marketing

### Background:

We are a small startup focusing mainly on providing machine learning solutions in the European banking market. We work on a variety of problems including fraud detection, sentiment classification and customer intention prediction and classification.

We are interested in developing a robust machine learning system that leverages information coming from call center data.

Ultimately, we are looking for ways to improve the success rate for calls made to customers for any product that our clients offer. Towards this goal we are working on designing an ever evolving machine learning product that offers high success outcomes while offering interpretability for our clients to make informed decisions.

### Goal(s):

Predict if the customer will subscribe (yes/no) to a term deposit (variable y)

Hit %81 or above accuracy by evaluating with 5-fold cross-validation and reporting the average performance score.

Find the customers who are more likely to buy the investment product. Determine the segment(s) of customers our client should prioritize. Find out What makes the customers buy? - Which feature should be the focuse be on.

### Data Description:

The data comes from the direct marketing efforts of a European banking institution. The marketing campaign involves making a phone call to a customer, often multiple times to ensure a product subscription, in this case, a term deposit. Term deposits are usually short-term deposits with maturities ranging from one month to a few years. When buying a term deposit, the customer must understand that they can withdraw their funds only after the term ends. All customer information that might reveal personal information is removed due to privacy concerns.

Attributes:

age : age of customer (numeric)

job : type of job (categorical)

marital : marital status (categorical)

education (categorical)

default: has credit in default? (binary)

balance: average yearly balance, in euros (numeric)

housing: has a housing loan? (binary)

loan: has a personal loan? (binary)

contact: contact communication type (categorical)

day: last contact day of the month (numeric)

month: last contact month of year (categorical)

duration: last contact duration, in seconds (numeric)

campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)

Output (desired target):

y - has the client subscribed to a term deposit? (binary)

### Methodology:

#### Part1: Exploratory Data Analysis(EDA):

1)Analysis of the features. Explored the data size, there were 40000 data point and data type, 4 of the features were numerical and the rest of the data features were categorical. In the data description, we could see that there was not null value at all and the target feature was very imbalanced. Customers who were not subscribe the term depoit were majority, this could come up with under-sampling issue when train the model.

<img width="971" alt="Screenshot 2024-10-04 at 10 40 37 AM" src="https://github.com/user-attachments/assets/9e6b5ab0-410a-40f5-aa86-2bfdec7da6e1">

<img width="342" alt="Screenshot 2024-10-04 at 10 41 00 AM" src="https://github.com/user-attachments/assets/16a1e139-e2d2-46c1-b21c-9faffbb23617">

<img width="248" alt="Screenshot 2024-10-04 at 10 41 18 AM" src="https://github.com/user-attachments/assets/2331944f-8380-4048-b5c4-cffeb96d99d0">

<img width="608" alt="Screenshot 2024-10-04 at 10 41 33 AM" src="https://github.com/user-attachments/assets/aed12df0-23e9-4cc5-9161-992df81f8865">

<img width="503" alt="Screenshot 2024-10-04 at 10 43 03 AM" src="https://github.com/user-attachments/assets/f2d4aa52-dc11-4c44-9865-87b521946365">

2)Finding any relations or trends considering multiple features.

<img width="879" alt="Screenshot 2024-10-04 at 12 38 23 PM" src="https://github.com/user-attachments/assets/c2a80813-ed1d-4ea0-8178-e8b8980011d0">

<img width="556" alt="Screenshot 2024-10-04 at 12 39 08 PM" src="https://github.com/user-attachments/assets/7b62cdf3-9b8a-47b0-a66a-541e3ec94c9b">

<img width="1236" alt="Screenshot 2024-10-04 at 12 39 28 PM" src="https://github.com/user-attachments/assets/8dd7c6cd-20ce-4e82-acc7-22a41ddec829">

<img width="1232" alt="Screenshot 2024-10-04 at 12 39 52 PM" src="https://github.com/user-attachments/assets/a2e324e0-9823-41f4-92f0-de35c4e961ad">


#### Part2: Feature Engineering and Data Cleaning:

1)Adding any few features. Not neccessary in this project.

2)Removing redundant features. In this step can reduce the dimensionality of the dataset by selecting the most informative features that contribute to the classification task. The result shows age, marital status, education, account balance, housing are the top 5 features will be used in predicted the target variable.

```
# Define features (x) and target (y)
x = data.drop('y', axis=1)
y = data['y']

# Perform feature selection
selector = SelectKBest(score_func=f_classif, k=10)
x_feature = selector.fit_transform(x, y)

# Get selected feature names
selected_features = x.columns[selector.get_support()]
print("Selected features:", selected_features)
```

3)Converting features into a suitable form for modeling by implemented Label Encoder function to convert categorical features into numerical.

#### Part3: Predictive Modeling

1)Running Basic Algorithms.

2)Cross Validation.

3)Ensembling.

4)Important Features Extraction.

5)The model performs very well on the customers do not subscribe the term deposit, but struggles significantly with the customers will subscribe the term deposit.

The high AUC-ROC indicates that the model is generally good at ranking positive and negative instances, but the low recall and F1-score for class 1 indicate that it is not effective at correctly identifying the minority class.

The confusion matrix highlights the issue with false negatives for class 1, where the model misses a substantial portion of actual positives. Like we mentioned earlier, the dataset has under-sampling issue, this is could be one of the reason of the model performance.

#### Part4: Model Training Pipeline

1)Set the pipeline, in order to solve the under-sampling problem, we want to try some resampling function and different model to make the best performance of the result. We set pipeline and train differen models under different sampling, feature selection techniques.

2)Set the parameter of GridResearch
```
# Define the pipeline with feature selection, resampling, and model training
pipeline = ImbPipeline([
    ('feature_selection', SelectKBest()),
    ('sampling', SMOTE(random_state=42)),
    ('model', RandomForestClassifier())
])

# Define the parameter grid for GridSearchCV
param_grid = {
    # Feature selection techniques
    'feature_selection': [
        SelectKBest(k=10),
        RFE(estimator=RandomForestClassifier(), n_features_to_select=10)
    ],

    # Resampling techniques
    'sampling': [
        SMOTE(random_state=42),
        RandomUnderSampler(random_state=42)
    ],

    # Model training with hyperparameters
    'model': [
        # Random Forest
        RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5),
        RandomForestClassifier(random_state=42, n_estimators=200, max_depth=10),

        # LightGBM
        lgb.LGBMClassifier(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=-1),
        lgb.LGBMClassifier(random_state=42, n_estimators=200, learning_rate=0.05, max_depth=-1),

        # XGBoost
        xgb.XGBClassifier(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=5, use_label_encoder=False, eval_metric='logloss'),
        xgb.XGBClassifier(random_state=42, n_estimators=200, learning_rate=0.05, max_depth=7, use_label_encoder=False, eval_metric='logloss')
    ]
}

# Set up cross-validation scheme
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Use GridSearchCV to search for the best combination of techniques
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# Fit the grid search to the training data
grid_search.fit(x_train, y_train)

# Get the best combination of techniques
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation F1 score: ", grid_search.best_score_)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)
print(classification_report(y_test, y_pred))
```

3)Evaluate the best model:Best parameters found:  {'feature_selection': RFE(estimator=RandomForestClassifier(), n_features_to_select=10), 'model': LGBMClassifier(learning_rate=0.05, n_estimators=200, random_state=42), 'sampling': SMOTE(random_state=42)}
Best cross-validation F1 score:  0.5005768957971746

F1-Score: 0.51: The F1-score for class 1 (customer would subscribe the term deposit) reflects the trade-off between precision and recall. A score of 0.51 indicates that, while the model does a decent job in identifying true positives, the precision issues reduce the overall effectiveness of class 1 predictions.

4)Model Explanation, use SHAP to explain the best model's predictions.

![image](https://github.com/user-attachments/assets/66f1df24-139f-4674-8c38-c151651fad20)

![image](https://github.com/user-attachments/assets/cadb7642-3d96-45a3-b2ee-3ecfe2a93cc3)

In the force plot showed that marital and campaign drive the negative impact to the subscription, age, account balance, last contact of month and education have contribute to the positive impact on subscription.

### Conclusion:

Based on the insights from the plot and table above, it is clear that “Duration” is a key feature in predicting customer outcomes. When prioritizing these features, “Duration”—the length of calls to the customer—should be the primary focus. Next in importance is “Balance,” which reflects the average yearly balance. Following that, attention should be given to the day of the month the customer is contacted, and then to the customer’s age, in that order.

<img width="260" alt="Screenshot 2024-10-04 at 12 38 56 PM" src="https://github.com/user-attachments/assets/7356dd99-ca8c-4131-ab2d-5c47767dd890">

<img width="287" alt="Screenshot 2024-10-04 at 1 14 22 PM" src="https://github.com/user-attachments/assets/4a6732c8-684d-41bc-8e30-286efa4763a0">


The high and medium segments, consisting of middle-aged, highly educated, high-net-worth customers without housing loans, should be the main focus for marketing term deposits. These segments show a higher likelihood of subscribing and are more willing to engage for longer periods during contact, which is a strong indicator of their intent to subscribe. Therefore, marketing efforts should prioritize these groups, limit campaigns to no more than two, and schedule them around the middle of the month during the first quarter of the year. This targeted strategy aims to optimize customer subscriptions to term deposits.
