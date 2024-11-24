---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region papermill={"duration": 0.010889, "end_time": "2024-01-30T02:10:04.314075", "exception": false, "start_time": "2024-01-30T02:10:04.303186", "status": "completed"} -->
# Introduction

Welcome to Binary Classification with Bank Churn Dataset competition! Here we will predict a customer churn based on their information. The dataset contains 13 columns:

1. Customer ID: A unique identifier for each customer
2. Surname: The customer's surname or last name
3. Credit Score: A numerical value representing the customer's credit score
4. Geography: The country where the customer resides (France, Spain or Germany)
5. Gender: The customer's gender (Male or Female)
6. Age: The customer's age.
7. Tenure: The number of years the customer has been with the bank
8. Balance: The customer's account balance
9. NumOfProducts: The number of bank products the customer uses (e.g., savings account, credit card)
10. HasCrCard: Whether the customer has a credit card (1 = yes, 0 = no)
11. IsActiveMember: Whether the customer is an active member (1 = yes, 0 = no)
12. EstimatedSalary: The estimated salary of the customer
13. Exited: Whether the customer has churned (1 = yes, 0 = no)

The metric we will use is Area Under the ROC Curve. If you want to read the description of the original dataset, you can visit this page: https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction
<!-- #endregion -->

<!-- #region papermill={"duration": 0.010726, "end_time": "2024-01-30T02:10:04.335407", "exception": false, "start_time": "2024-01-30T02:10:04.324681", "status": "completed"} -->
# Loading Libraries and Datasets
<!-- #endregion -->

```python _kg_hide-input=false _kg_hide-output=true papermill={"duration": 18.411988, "end_time": "2024-01-30T02:10:22.758176", "exception": false, "start_time": "2024-01-30T02:10:04.346188", "status": "completed"}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import optuna

from category_encoders import OneHotEncoder, MEstimateEncoder, CatBoostEncoder, OrdinalEncoder
from sklearn import set_config
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import roc_auc_score, roc_curve, make_scorer, f1_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.preprocessing import FunctionTransformer, StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, RidgeClassifierCV
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from xgboost import XGBClassifier, XGBRFClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

sns.set_theme(style = 'white', palette = 'viridis')
pal = sns.color_palette('viridis')

pd.set_option('display.max_rows', 100)
set_config(transform_output = 'pandas')
pd.options.mode.chained_assignment = None
```

```python _kg_hide-input=false papermill={"duration": 0.566205, "end_time": "2024-01-30T02:10:23.334711", "exception": false, "start_time": "2024-01-30T02:10:22.768506", "status": "completed"}
train = pd.read_csv(r'/kaggle/input/playground-series-s4e1/train.csv', index_col = 'id').astype({'IsActiveMember' : np.uint8, 'HasCrCard' : np.uint8})
test = pd.read_csv(r'/kaggle/input/playground-series-s4e1/test.csv', index_col = 'id').astype({'IsActiveMember' : np.uint8, 'HasCrCard' : np.uint8})
orig_train = pd.read_csv(r'/kaggle/input/bank-customer-churn-prediction/Churn_Modelling.csv', index_col = 'RowNumber')
```

<!-- #region papermill={"duration": 0.010461, "end_time": "2024-01-30T02:10:23.355659", "exception": false, "start_time": "2024-01-30T02:10:23.345198", "status": "completed"} -->
# Descriptive Statistics

Let's begin by taking a peek at our training dataset first
<!-- #endregion -->

```python _kg_hide-input=true papermill={"duration": 0.037107, "end_time": "2024-01-30T02:10:23.403288", "exception": false, "start_time": "2024-01-30T02:10:23.366181", "status": "completed"}
train.head(10)
```

```python _kg_hide-input=true papermill={"duration": 0.17629, "end_time": "2024-01-30T02:10:23.590460", "exception": false, "start_time": "2024-01-30T02:10:23.414170", "status": "completed"}
desc = pd.DataFrame(index = list(train))
desc['type'] = train.dtypes
desc['count'] = train.count()
desc['nunique'] = train.nunique()
desc['%unique'] = desc['nunique'] / len(train) * 100
desc['null'] = train.isnull().sum()
desc['%null'] = desc['null'] / len(train) * 100
desc['min'] = train.min()
desc['max'] = train.max()
desc
```

<!-- #region papermill={"duration": 0.010739, "end_time": "2024-01-30T02:10:23.612313", "exception": false, "start_time": "2024-01-30T02:10:23.601574", "status": "completed"} -->
We can see that we have 165k rows and 13 columns, including our target here, which makes it 12 features, with at least 3 of them being categorical.

Let's see the test dataset now.
<!-- #endregion -->

```python _kg_hide-input=true papermill={"duration": 0.029698, "end_time": "2024-01-30T02:10:23.652940", "exception": false, "start_time": "2024-01-30T02:10:23.623242", "status": "completed"}
test.head(10)
```

```python _kg_hide-input=true papermill={"duration": 0.124859, "end_time": "2024-01-30T02:10:23.789108", "exception": false, "start_time": "2024-01-30T02:10:23.664249", "status": "completed"}
desc = pd.DataFrame(index = list(test))
desc['type'] = test.dtypes
desc['count'] = test.count()
desc['nunique'] = test.nunique()
desc['%unique'] = desc['nunique'] / len(test) * 100
desc['null'] = test.isnull().sum()
desc['%null'] = desc['null'] / len(test) * 100
desc['min'] = test.min()
desc['max'] = test.max()
desc
```

<!-- #region papermill={"duration": 0.011614, "end_time": "2024-01-30T02:10:23.813226", "exception": false, "start_time": "2024-01-30T02:10:23.801612", "status": "completed"} -->
On the test dataset, we have 110k rows. There is also no missing value on both.

Finally, let's try to see the original dataset.
<!-- #endregion -->

```python _kg_hide-input=true papermill={"duration": 0.029373, "end_time": "2024-01-30T02:10:23.908115", "exception": false, "start_time": "2024-01-30T02:10:23.878742", "status": "completed"}
orig_train.head(10)
```

```python _kg_hide-input=true papermill={"duration": 0.047824, "end_time": "2024-01-30T02:10:23.967679", "exception": false, "start_time": "2024-01-30T02:10:23.919855", "status": "completed"}
desc = pd.DataFrame(index = list(orig_train))
desc['type'] = orig_train.dtypes
desc['count'] = orig_train.count()
desc['nunique'] = orig_train.nunique()
desc['%unique'] = desc['nunique'] / len(orig_train) * 100
desc['null'] = orig_train.isnull().sum()
desc['%null'] = desc['null'] / len(orig_train) * 100
desc['min'] = orig_train.min()
desc['max'] = orig_train.max()
desc
```

```python papermill={"duration": 0.028231, "end_time": "2024-01-30T02:10:24.008186", "exception": false, "start_time": "2024-01-30T02:10:23.979955", "status": "completed"}
numerical_features = list(test._get_numeric_data())
categorical_features = list(test.drop(numerical_features, axis = 1))
```

<!-- #region papermill={"duration": 0.012148, "end_time": "2024-01-30T02:10:24.033124", "exception": false, "start_time": "2024-01-30T02:10:24.020976", "status": "completed"} -->
# Preparation

This is where we start preparing everything if we want to start building machine learning models.
<!-- #endregion -->

```python papermill={"duration": 0.347747, "end_time": "2024-01-30T02:10:24.393527", "exception": false, "start_time": "2024-01-30T02:10:24.045780", "status": "completed"}
X = pd.concat([orig_train, train]).reset_index(drop = True)
y = X.pop('Exited')

orig_comp_combo = train.merge(orig_train, on = list(test), how = 'left')
orig_comp_combo.index = train.index

orig_test_combo = test.merge(orig_train, on = list(test), how = 'left')
orig_test_combo.index = test.index

seed = 42
splits = 30
skf = StratifiedKFold(n_splits = splits, random_state = seed, shuffle = True)
tf.keras.utils.set_random_seed(seed)
tf.config.experimental.enable_op_determinism()
```

<!-- #region papermill={"duration": 0.012655, "end_time": "2024-01-30T02:10:24.418832", "exception": false, "start_time": "2024-01-30T02:10:24.406177", "status": "completed"} -->
# Feature Engineering
<!-- #endregion -->

```python papermill={"duration": 0.022589, "end_time": "2024-01-30T02:10:24.453702", "exception": false, "start_time": "2024-01-30T02:10:24.431113", "status": "completed"}
def nullify(x):
    x_copy = x.copy()
    x_copy['Balance'] = x_copy['Balance'].replace({0 : np.nan})
    return x_copy

Nullify = FunctionTransformer(nullify)
```

```python papermill={"duration": 0.020408, "end_time": "2024-01-30T02:10:24.486588", "exception": false, "start_time": "2024-01-30T02:10:24.466180", "status": "completed"}
def salary_rounder(x):
    x_copy = x.copy()
    x_copy['EstimatedSalary'] = (x_copy['EstimatedSalary'] * 100).astype(np.uint64)
    return x_copy

SalaryRounder = FunctionTransformer(salary_rounder)
```

```python papermill={"duration": 0.023741, "end_time": "2024-01-30T02:10:24.522889", "exception": false, "start_time": "2024-01-30T02:10:24.499148", "status": "completed"}
def age_rounder(x):
    x_copy = x.copy()
    x_copy['Age'] = (x_copy['Age'] * 10).astype(np.uint16)
    return x_copy

AgeRounder = FunctionTransformer(age_rounder)
```

```python papermill={"duration": 0.020722, "end_time": "2024-01-30T02:10:24.556222", "exception": false, "start_time": "2024-01-30T02:10:24.535500", "status": "completed"}
def balance_rounder(x):
    x_copy = x.copy()
    x_copy['Balance'] = (x_copy['Balance'] * 100).astype(np.uint64)
    return x_copy

BalanceRounder = FunctionTransformer(balance_rounder)
```

```python papermill={"duration": 0.021601, "end_time": "2024-01-30T02:10:24.590693", "exception": false, "start_time": "2024-01-30T02:10:24.569092", "status": "completed"}
def feature_generator(x):
    
    x_copy = x.copy()
    #x_copy['IsSenior'] = (x_copy['Age'] >= 600).astype(np.uint8)
    x_copy['IsActive_by_CreditCard'] = x_copy['HasCrCard'] * x_copy['IsActiveMember']
    x_copy['Products_Per_Tenure'] =  x_copy['Tenure'] / x_copy['NumOfProducts']
    x_copy['ZeroBalance'] = (x_copy['Balance'] == 0).astype(np.uint8)
    x_copy['AgeCat'] = np.round(x_copy.Age/20).astype(np.uint16)#.astype('category')
    x_copy['AllCat'] = x_copy['Surname']+x_copy['Geography']+x_copy['Gender']+x_copy.EstimatedSalary.astype('str')+x_copy.CreditScore.astype('str')+x_copy.Age.astype('str')+x_copy.NumOfProducts.astype('str')+x_copy.Tenure.astype('str')+x_copy.CustomerId.astype('str')#+np.round(x_copy.IsActiveMember).astype('str')
    
    return x_copy

FeatureGenerator = FunctionTransformer(feature_generator)
```

```python papermill={"duration": 0.021665, "end_time": "2024-01-30T02:10:24.624852", "exception": false, "start_time": "2024-01-30T02:10:24.603187", "status": "completed"}
def svd_rounder(x):
    
    x_copy = x.copy()
    for col in [column for column in list(x) if 'SVD' in column]:
        x_copy[col] = (x_copy[col] * 1e18).astype(np.int64)
        
    return x_copy

SVDRounder = FunctionTransformer(svd_rounder)
```

```python papermill={"duration": 0.020475, "end_time": "2024-01-30T02:10:24.657878", "exception": false, "start_time": "2024-01-30T02:10:24.637403", "status": "completed"}
class FeatureDropper(BaseEstimator, TransformerMixin):
    
    def __init__(self, cols):
        self.cols = cols
        
    def fit(self, x, y):
        return self
    
    def transform(self, x):
        return x.drop(self.cols, axis = 1)
```

```python papermill={"duration": 0.020025, "end_time": "2024-01-30T02:10:24.690461", "exception": false, "start_time": "2024-01-30T02:10:24.670436", "status": "completed"}
class Categorizer(BaseEstimator, TransformerMixin):
    
    def __init__(self, cols : list):
        self.cols = cols
        
    def fit(self, x, y):
        return self
    
    def transform(self, x):
        return x.astype({cat : 'category' for cat in self.cols})
```

```python papermill={"duration": 0.023421, "end_time": "2024-01-30T02:10:24.726792", "exception": false, "start_time": "2024-01-30T02:10:24.703371", "status": "completed"}
class Vectorizer(BaseEstimator, TransformerMixin):
    
    def __init__(self, max_features = 1000, cols = ['Surname'], n_components = 3):
        self.max_features = max_features
        self.cols = cols
        self.n_components = n_components
        
    def fit(self, x, y):
        self.vectorizer_dict = {}
        self.decomposer_dict = {}
        
        for col in self.cols:
            self.vectorizer_dict[col] = TfidfVectorizer(max_features = self.max_features).fit(x[col].astype(str), y)
            self.decomposer_dict[col] = TruncatedSVD(random_state = seed, n_components = self.n_components).fit(
                self.vectorizer_dict[col].transform(x[col].astype(str)), y
            )
        
        return self
    
    def transform(self, x):
        vectorized = {}
        
        for col in self.cols:
            vectorized[col] = self.vectorizer_dict[col].transform(x[col].astype(str))
            vectorized[col] = self.decomposer_dict[col].transform(vectorized[col])
        
        vectorized_df = pd.concat([pd.DataFrame(vectorized[col]).rename({
            f'truncatedsvd{i}' : f'{col}SVD{i}' for i in range(self.n_components)
        }, axis = 1) for col in self.cols], axis = 1)
        
        return pd.concat([x.reset_index(drop = True), vectorized_df], axis = 1)
```

<!-- #region papermill={"duration": 0.01261, "end_time": "2024-01-30T02:10:24.751951", "exception": false, "start_time": "2024-01-30T02:10:24.739341", "status": "completed"} -->
# Model Cross Validation

Let's start by evaluating the performance of our model first. We will use M-Estimate Encoder and Standard Scaler in our pipeline. We will also concatenate the original dataset only during the cross-validation process for robustness.
<!-- #endregion -->

```python _kg_hide-input=false papermill={"duration": 0.02928, "end_time": "2024-01-30T02:10:24.793861", "exception": false, "start_time": "2024-01-30T02:10:24.764581", "status": "completed"}
def cross_val_score(estimator, cv = skf, label = '', include_original = True, show_importance = False, add_reverse = False):
    
    X = train.copy()
    y = X.pop('Exited')
    
    #initiate prediction arrays and score lists
    val_predictions = np.zeros((len(X)))
    train_scores, val_scores= [], []
    
    feature_importances_table = pd.DataFrame({'value' : 0}, index = list(X.columns))
    
    test_predictions = np.zeros((len(test)))
    
    #training model, predicting prognosis probability, and evaluating metrics
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        
        model = clone(estimator)
        
        #define train set
        X_train = X.iloc[train_idx].reset_index(drop = True)
        y_train = y.iloc[train_idx].reset_index(drop = True)
        
        #define validation set
        X_val = X.iloc[val_idx].reset_index(drop = True)
        y_val = y.iloc[val_idx].reset_index(drop = True)
        
        if include_original:
            X_train = pd.concat([orig_train.drop('Exited', axis = 1), X_train]).reset_index(drop = True)
            y_train = pd.concat([orig_train.Exited, y_train]).reset_index(drop = True)
            
        if add_reverse:
            X_train = pd.concat([X_train, X_train.iloc[::-1]]).reset_index(drop = True)
            y_train = pd.concat([y_train, y_train.iloc[::-1]]).reset_index(drop = True)
        
        #train model
        model.fit(X_train, y_train)
        
        #make predictions
        train_preds = model.predict_proba(X_train)[:, 1]
        val_preds = model.predict_proba(X_val)[:, 1]
                  
        val_predictions[val_idx] += val_preds
        test_predictions += model.predict_proba(test)[:, 1] / cv.get_n_splits()
        if show_importance:
            feature_importances_table['value'] += permutation_importance(model, X_val, y_val, random_state = seed, scoring = make_scorer(roc_auc_score, needs_proba = True), n_repeats = 5).importances_mean / cv.get_n_splits()
        
        #evaluate model for a fold
        train_score = roc_auc_score(y_train, train_preds)
        val_score = roc_auc_score(y_val, val_preds)
        
        #print(f'Fold {fold}: {val_score:.5f}')
        
        #append model score for a fold to list
        train_scores.append(train_score)
        val_scores.append(val_score)
       
    if show_importance:
        plt.figure(figsize = (20, 30))
        plt.title(f'Features with Biggest Importance of {np.mean(val_scores):.5f} ± {np.std(val_scores):.5f} Model', size = 25, weight = 'bold')
        sns.barplot(feature_importances_table.sort_values('value', ascending = False).T, orient = 'h', palette = 'viridis')
        plt.show()
    else:
        print(f'Val Score: {np.mean(val_scores):.5f} ± {np.std(val_scores):.5f} | Train Score: {np.mean(train_scores):.5f} ± {np.std(train_scores):.5f} | {label}')
        
    val_predictions = np.where(orig_comp_combo.Exited_y == 1, 0, np.where(orig_comp_combo.Exited_y == 0, 1, val_predictions))
    test_predictions = np.where(orig_test_combo.Exited == 1, 0, np.where(orig_test_combo.Exited == 0, 1, test_predictions))
    
    return val_scores, val_predictions, test_predictions
```

```python papermill={"duration": 0.021453, "end_time": "2024-01-30T02:10:24.827695", "exception": false, "start_time": "2024-01-30T02:10:24.806242", "status": "completed"}
score_list, oof_list, predict_list = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

cat_features = ['CustomerId', 'Surname', 'EstimatedSalary', 'Geography', 'Gender', 'Tenure', 'Age', 'NumOfProducts', 'IsActiveMember', 'CreditScore', 'AllCat', 'IsActive_by_CreditCard']
```

<!-- #region papermill={"duration": 0.01163, "end_time": "2024-01-30T02:10:24.851636", "exception": false, "start_time": "2024-01-30T02:10:24.840006", "status": "completed"} -->
# Logistic Regression
<!-- #endregion -->

```python papermill={"duration": 616.017179, "end_time": "2024-01-30T02:20:40.880824", "exception": false, "start_time": "2024-01-30T02:10:24.863645", "status": "completed"}
Log = make_pipeline(
    SalaryRounder,
    AgeRounder,
    FeatureGenerator,
    Vectorizer(cols = ['Surname', 'AllCat', 'EstimatedSalary', 'CreditScore'], max_features = 500, n_components = 4),
    CatBoostEncoder(cols = cat_features + [f'SurnameSVD{i}' for i in range(4)]),# + [f'AllCatSVD{i}' for i in range(6)] + [f'EstimatedSalarySVD{i}' for i in range(6)] + [f'CreditScoreSVD{i}' for i in range(6)]),
    StandardScaler(),
    LogisticRegression(random_state = seed, max_iter = 1000000000)
)

_, oof_list['Log'], predict_list['Log'] = cross_val_score(Log)
```

<!-- #region papermill={"duration": 0.012221, "end_time": "2024-01-30T02:20:40.905634", "exception": false, "start_time": "2024-01-30T02:20:40.893413", "status": "completed"} -->
# Tensorflow
<!-- #endregion -->

```python papermill={"duration": 0.02626, "end_time": "2024-01-30T02:20:40.944751", "exception": false, "start_time": "2024-01-30T02:20:40.918491", "status": "completed"}
class TensorFlower(BaseEstimator, ClassifierMixin):
    
    def fit(self, x, y):
        inputs = tf.keras.Input((x.shape[1]))
        inputs_norm = tf.keras.layers.BatchNormalization()(inputs)
        
        z = tf.keras.layers.Dense(32)(inputs_norm)
        z = tf.keras.layers.BatchNormalization()(z)
        z = tf.keras.layers.LeakyReLU()(z)
        #z = tf.keras.layers.Dropout(.4)(z)
        
        z = tf.keras.layers.Dense(64)(z)
        z = tf.keras.layers.BatchNormalization()(z)
        z = tf.keras.layers.LeakyReLU()(z)
        #z = tf.keras.layers.Dropout(.4)(z)
        
        z = tf.keras.layers.Dense(16)(z)
        z = tf.keras.layers.BatchNormalization()(z)
        z = tf.keras.layers.LeakyReLU()(z)
        #z = tf.keras.layers.Dropout(.4)(z)
        
        z = tf.keras.layers.Dense(4)(z)
        z = tf.keras.layers.BatchNormalization()(z)
        z = tf.keras.layers.LeakyReLU()(z)
        #z = tf.keras.layers.Dropout(.4)(z)
        
        z = tf.keras.layers.Dense(1)(z)
        z = tf.keras.layers.BatchNormalization()(z)
        outputs = tf.keras.activations.sigmoid(z)
        
        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(loss = 'binary_crossentropy', optimizer = tf.keras.optimizers.AdamW(1e-4))
        
        self.model.fit(x.to_numpy(), y, epochs = 10, verbose = 0)
        self.classes_ = np.unique(y)
        
        return self
    def predict_proba(self, x):
        predictions = np.zeros((len(x), 2))
        predictions[:, 1] = self.model.predict(x, verbose = 0)[:, 0]
        predictions[:, 0] = 1 - predictions[:, 1]
        return predictions
    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis = 1)
```

```python papermill={"duration": 5517.525611, "end_time": "2024-01-30T03:52:38.527597", "exception": false, "start_time": "2024-01-30T02:20:41.001986", "status": "completed"}
TensorFlowey = make_pipeline(
    SalaryRounder,
    AgeRounder,
    FeatureGenerator,
    #Vectorizer(cols = ['Surname', 'AllCat', 'EstimatedSalary', 'CreditScore'], max_features = 500, n_components = 6),
    CatBoostEncoder(cols = cat_features),
    TensorFlower()
)

_, oof_list['TF'], predict_list['TF'] = cross_val_score(TensorFlowey)
```

<!-- #region papermill={"duration": 0.011834, "end_time": "2024-01-30T03:52:38.552519", "exception": false, "start_time": "2024-01-30T03:52:38.540685", "status": "completed"} -->
# XGBoost
<!-- #endregion -->

```python papermill={"duration": 0.024363, "end_time": "2024-01-30T03:52:38.589065", "exception": false, "start_time": "2024-01-30T03:52:38.564702", "status": "completed"}
def xgb_objective(trial):
    params = {
        'eta' : trial.suggest_float('eta', .001, .3, log = True),
        'max_depth' : trial.suggest_int('max_depth', 2, 30),
        'subsample' : trial.suggest_float('subsample', .5, 1),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', .1, 1),
        'min_child_weight' : trial.suggest_float('min_child_weight', .1, 20, log = True),
        'reg_lambda' : trial.suggest_float('reg_lambda', .01, 20, log = True),
        'reg_alpha' : trial.suggest_float('reg_alpha', .01, 10, log = True),
        'n_estimators' : 1000,
        'random_state' : seed,
        'tree_method' : 'hist',
    }
    
    optuna_model = make_pipeline(
        SalaryRounder,
        AgeRounder,
        FeatureGenerator,
        Vectorizer(cols = ['Surname', 'AllCat', 'EstimatedSalary', 'CustomerId'], max_features = 1000, n_components = 3),
        CatBoostEncoder(cols = ['CustomerId', 'Surname', 'EstimatedSalary', 'AllCat', 'CreditScore']),
        MEstimateEncoder(cols = ['Geography', 'Gender']),
        XGBClassifier(**params)
    )
    
    optuna_score, _, _ = cross_val_score(optuna_model)
    
    return np.mean(optuna_score)

xgb_study = optuna.create_study(direction = 'maximize')
```

```python papermill={"duration": 0.019028, "end_time": "2024-01-30T03:52:38.620257", "exception": false, "start_time": "2024-01-30T03:52:38.601229", "status": "completed"}
#xgb_study.optimize(xgb_objective, 50)
```

```python papermill={"duration": 1118.30541, "end_time": "2024-01-30T04:11:16.937883", "exception": false, "start_time": "2024-01-30T03:52:38.632473", "status": "completed"}
xgb_params = {'eta': 0.04007938900538817, 'max_depth': 5, 'subsample': 0.8858539721226424, 'colsample_bytree': 0.41689519430449395, 'min_child_weight': 0.4225662401139526, 'reg_lambda': 1.7610231110037127, 'reg_alpha': 1.993860687732973}

XGB = make_pipeline(
    SalaryRounder,
    AgeRounder,
    FeatureGenerator,
    Vectorizer(cols = ['Surname', 'AllCat', 'EstimatedSalary', 'CustomerId'], max_features = 1000, n_components = 3),
    CatBoostEncoder(cols = ['CustomerId', 'Surname', 'EstimatedSalary', 'AllCat', 'CreditScore']),
    MEstimateEncoder(cols = ['Geography', 'Gender']),
    XGBClassifier(**xgb_params, random_state = seed, tree_method = 'hist', n_estimators = 1000)
)

_, oof_list['XGB'], predict_list['XGB'] = cross_val_score(XGB, show_importance = False)
```

<!-- #region papermill={"duration": 0.013014, "end_time": "2024-01-30T04:11:16.963723", "exception": false, "start_time": "2024-01-30T04:11:16.950709", "status": "completed"} -->
# LightGBM
<!-- #endregion -->

```python papermill={"duration": 0.026091, "end_time": "2024-01-30T04:11:17.002339", "exception": false, "start_time": "2024-01-30T04:11:16.976248", "status": "completed"}
def lgb_objective(trial):
    params = {
        'learning_rate' : trial.suggest_float('learning_rate', .001, .1, log = True),
        'max_depth' : trial.suggest_int('max_depth', 2, 20),
        'subsample' : trial.suggest_float('subsample', .5, 1),
        'min_child_weight' : trial.suggest_float('min_child_weight', .1, 15, log = True),
        'reg_lambda' : trial.suggest_float('reg_lambda', .1, 20, log = True),
        'reg_alpha' : trial.suggest_float('reg_alpha', .1, 10, log = True),
        'n_estimators' : 1000,
        'random_state' : seed,
        #'boosting_type' : 'dart',
    }
    
    optuna_model = make_pipeline(
        SalaryRounder,
        AgeRounder,
        FeatureGenerator,
        Vectorizer(cols = ['Surname', 'AllCat'], max_features = 1000, n_components = 3),
        CatBoostEncoder(cols = ['Surname', 'AllCat', 'CreditScore', 'Age']),
        MEstimateEncoder(cols = ['Geography', 'Gender', 'NumOfProducts']),
        StandardScaler(),
        LGBMClassifier(**params)
    )
    
    optuna_score, _, _ = cross_val_score(optuna_model)
    
    return np.mean(optuna_score)

lgb_study = optuna.create_study(direction = 'maximize')
```

```python papermill={"duration": 0.021904, "end_time": "2024-01-30T04:11:17.036996", "exception": false, "start_time": "2024-01-30T04:11:17.015092", "status": "completed"}
#lgb_study.optimize(lgb_objective, 100)
```

```python papermill={"duration": 1293.231267, "end_time": "2024-01-30T04:32:50.282626", "exception": false, "start_time": "2024-01-30T04:11:17.051359", "status": "completed"}
lgb_params = {'learning_rate': 0.01864960338160943, 'max_depth': 9, 'subsample': 0.6876252164703066, 'min_child_weight': 0.8117588782708633, 'reg_lambda': 6.479178739677389, 'reg_alpha': 3.2952573115561234}

LGB = make_pipeline(
    SalaryRounder,
    AgeRounder,
    FeatureGenerator,
    Vectorizer(cols = ['Surname', 'AllCat'], max_features = 1000, n_components = 3),
    CatBoostEncoder(cols = ['Surname', 'AllCat', 'CreditScore', 'Age']),
    MEstimateEncoder(cols = ['Geography', 'Gender', 'NumOfProducts']),
    StandardScaler(),
    LGBMClassifier(**lgb_params, random_state = seed, n_estimators = 1000)
)

_, oof_list['LGB'], predict_list['LGB'] = cross_val_score(LGB, show_importance = False)
```

<!-- #region papermill={"duration": 0.0131, "end_time": "2024-01-30T04:32:50.309573", "exception": false, "start_time": "2024-01-30T04:32:50.296473", "status": "completed"} -->
# CatBoost
<!-- #endregion -->

```python papermill={"duration": 10170.159848, "end_time": "2024-01-30T07:22:20.482501", "exception": false, "start_time": "2024-01-30T04:32:50.322653", "status": "completed"}
CB = make_pipeline(
    SalaryRounder,
    AgeRounder,
    FeatureGenerator,
    Vectorizer(cols = ['Surname', 'AllCat'], max_features = 1000, n_components = 4),
    SVDRounder,
    CatBoostClassifier(random_state = seed, verbose = 0, cat_features = cat_features + [f'SurnameSVD{i}' for i in range(4)], has_time = True)
)

_, oof_list['CB'], predict_list['CB'] = cross_val_score(CB, show_importance = False)
```

```python papermill={"duration": 10889.115027, "end_time": "2024-01-30T10:23:49.610098", "exception": false, "start_time": "2024-01-30T07:22:20.495071", "status": "completed"}
CB_Bayes = make_pipeline(
    SalaryRounder,
    AgeRounder,
    FeatureGenerator,
    Vectorizer(cols = ['Surname', 'AllCat'], max_features = 1000, n_components = 4),
    SVDRounder,
    CatBoostClassifier(random_state = seed, verbose = 0, cat_features = cat_features + [f'SurnameSVD{i}' for i in range(4)], bootstrap_type = 'Bayesian', has_time = True)
)

_, oof_list['CB_Bayes'], predict_list['CB_Bayes'] = cross_val_score(CB_Bayes, show_importance = False)
```

```python papermill={"duration": 10176.469522, "end_time": "2024-01-30T13:13:26.093152", "exception": false, "start_time": "2024-01-30T10:23:49.623630", "status": "completed"}
CB_Bernoulli = make_pipeline(
    SalaryRounder,
    AgeRounder,
    FeatureGenerator,
    Vectorizer(cols = ['Surname', 'AllCat'], max_features = 1000, n_components = 4),
    SVDRounder,
    CatBoostClassifier(random_state = seed, verbose = 0, cat_features = cat_features + [f'SurnameSVD{i}' for i in range(4)], bootstrap_type = 'Bernoulli', has_time = True)
)

_, oof_list['CB_Bernoulli'], predict_list['CB_Bernoulli'] = cross_val_score(CB_Bernoulli, show_importance = False)
```

<!-- #region papermill={"duration": 0.012707, "end_time": "2024-01-30T13:13:26.263247", "exception": false, "start_time": "2024-01-30T13:13:26.250540", "status": "completed"} -->
# Voting Ensemble
<!-- #endregion -->

```python papermill={"duration": 0.076089, "end_time": "2024-01-30T13:13:26.353603", "exception": false, "start_time": "2024-01-30T13:13:26.277514", "status": "completed"}
weights = RidgeClassifier(random_state = seed).fit(oof_list, train.Exited).coef_[0]
weights /= weights.sum()
pd.DataFrame(weights, index = list(oof_list), columns = ['weight per model'])
```

```python papermill={"duration": 0.091348, "end_time": "2024-01-30T13:13:26.461860", "exception": false, "start_time": "2024-01-30T13:13:26.370512", "status": "completed"}
#_, ensemble_oof, predictions = cross_val_score(voter, show_importance = False)
print(f'Score: {(roc_auc_score(train.Exited, oof_list.to_numpy() @ weights)):.5f}')
predictions = predict_list.to_numpy() @ weights
```

<!-- #region papermill={"duration": 0.018992, "end_time": "2024-01-30T13:13:26.498650", "exception": false, "start_time": "2024-01-30T13:13:26.479658", "status": "completed"} -->
# Submission
<!-- #endregion -->

```python papermill={"duration": 0.263108, "end_time": "2024-01-30T13:13:26.779363", "exception": false, "start_time": "2024-01-30T13:13:26.516255", "status": "completed"}
submission = test.copy()
submission['Exited'] = np.where(orig_test_combo.Exited == 1, 0, np.where(orig_test_combo.Exited == 0, 1, predictions))

submission.Exited.to_csv('submission.csv')
```

```python _kg_hide-input=true papermill={"duration": 1.558689, "end_time": "2024-01-30T13:13:28.351894", "exception": false, "start_time": "2024-01-30T13:13:26.793205", "status": "completed"}
plt.figure(figsize = (15, 10), dpi = 300)
sns.kdeplot(submission.Exited, fill = True)
plt.title("Distribution of Customer Churn Probability", weight = 'bold', size = 25)
plt.show()
```

<!-- #region papermill={"duration": 0.015429, "end_time": "2024-01-30T13:13:28.383227", "exception": false, "start_time": "2024-01-30T13:13:28.367798", "status": "completed"} -->
Thanks for reading!
<!-- #endregion -->
