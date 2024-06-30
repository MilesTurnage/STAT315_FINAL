```python
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from IPython.core.interactiveshell import InteractiveShell
import statsmodels.api as sm 
from mlxtend.evaluate import bias_variance_decomp
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

InteractiveShell.ast_node_interactivity = "all"
```


```python
file_path = 'customer_booking.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Exercise I: Preparing the dataset
# Filter the dataset for Internet sales channel (excluding Mobile) and Round Trip (excluding Circle and One-Way Trips)
filtered_df = df[(df['sales_channel'] == 'Internet') & (df['trip_type'] == 'RoundTrip')]

# Select relevant columns and rename them
filtered_df = filtered_df[['num_passengers', 'purchase_lead', 'booking_complete']]
filtered_df = filtered_df.rename(columns={'num_passengers': 'size', 'purchase_lead': 'lead_time', 'booking_complete': 'booking'})

# Convert to NumPy dataframe
numpy_df = filtered_df.to_numpy()
```


```python
#Exercise II: Visualizing the data.
#plot data
sns.scatterplot(data=filtered_df, x='size', y='lead_time', hue='booking', alpha=0.5)
```




    <Axes: xlabel='size', ylabel='lead_time'>






    
![png](final_files/final_2_1.png)
    



Exercise III: Preliminary analysis:




```python
X = filtered_df[['size', 'lead_time']]
y = filtered_df['booking']

# Exercise IV: Cross validation


"""
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
max_depthint, default=None
The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
"""

# Logistic Regression
log_reg = LogisticRegression(max_iter=100)
log_reg = log_reg.fit(X, y)
log_reg_scores = cross_val_score(log_reg, X, y, cv=10)

# Decision Tree Classifier
tree_clf = tree.DecisionTreeClassifier(max_depth=10)
tree_clf = tree_clf.fit(X, y)
tree_clf_scores = cross_val_score(tree_clf, X, y, cv=10)

# Gaussian Naive Bayes
gnb = GaussianNB()
gnb = gnb.fit(X, y)
gnb_scores = cross_val_score(gnb, X, y, cv=10)

# Output the mean scores for comparison
print(f'Logistic Regression Mean CV Score: {log_reg_scores.mean()}')
print(f'Decision Tree Classifier Mean CV Score: {tree_clf_scores.mean()}')
print(f'Gaussian Naive Bayes Mean CV Score: {gnb_scores.mean()}')
```




    '\nhttps://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html\nmax_depthint, default=None\nThe maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.\n'



    Logistic Regression Mean CV Score: 0.8440582232553748
    Decision Tree Classifier Mean CV Score: 0.8429648447409255
    Gaussian Naive Bayes Mean CV Score: 0.8414614175091005



```python
#Exercise V: Optimal tree depth

#find optimal depth and plot tree

dpth = np.linspace(1, 50, 50).astype(int)
scores = np.zeros(len(dpth))

for i in dpth:
    clf = tree.DecisionTreeClassifier(max_depth=i)
    clf = clf.fit(X, y)
    scores[i-1] = cross_val_score(clf, X, y, cv=10).mean()
    
plt.plot(dpth, scores)
print(f'Optimal tree depth: {dpth[np.argmax(scores)]}')


optimal_depth = dpth[np.argmax(scores)]
#optimal_tree_clf = tree.DecisionTreeClassifier(max_depth=optimal_depth)
#optimal_tree_clf.fit(X, y)
#tree.plot_tree(optimal_tree_clf,feature_names=list(df.columns)[0:2])
```




    [<matplotlib.lines.Line2D at 0x7f4f7e1d3e20>]



    Optimal tree depth: 1





    
![png](final_files/final_5_2.png)
    




```python
#Exercise VI: Interpretation of the logit results

X = filtered_df[['size', 'lead_time']]
y = filtered_df['booking']

logit_model = sm.Logit(y, X)
result = logit_model.fit()

# Print the summary of the model
print(result.summary())

# Optionally, evaluate the model using cross-validation (sklearn)
log_reg = LogisticRegression(max_iter=100)
log_reg = log_reg.fit(X, y)
log_reg_scores = cross_val_score(log_reg, X, y, cv=10)

# Output the mean score for cross-validation
print(f'Logistic Regression Mean CV Score: {log_reg_scores.mean()}')
```

    Optimization terminated successfully.
             Current function value: 0.481773
             Iterations 6
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:                booking   No. Observations:                43901
    Model:                          Logit   Df Residuals:                    43899
    Method:                           MLE   Df Model:                            1
    Date:                Sun, 30 Jun 2024   Pseudo R-squ.:                 -0.1130
    Time:                        21:57:26   Log-Likelihood:                -21150.
    converged:                       True   LL-Null:                       -19004.
    Covariance Type:            nonrobust   LLR p-value:                     1.000
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    size          -0.7495      0.012    -64.330      0.000      -0.772      -0.727
    lead_time     -0.0038      0.000    -22.501      0.000      -0.004      -0.003
    ==============================================================================
    Logistic Regression Mean CV Score: 0.8440582232553748



```python
hnfrom mlxtend.evaluate import bias_variance_decomp

X = filtered_df[['size', 'lead_time']]
y = filtered_df['booking']

#Exercise VII: Statistical learning with DecisionTreeClassifier
"""
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
arrays
sequence of indexables with same length / shape[0]
Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.

test_size
float or int, default=None
If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25.

train_size
float or int, default=None
If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split. If int, represents the absolute number of train samples. If None, the value is automatically set to the complement of the test size.

random_state
int, RandomState instance or None, default=None
Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls. See Glossary.

shuffle
bool, default=True
Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.

stratify
array-like, default=None
If not None, data is split in a stratified fashion, using this as the class labels.
"""

# Split the data into a training (70%) and test data (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, shuffle=True, stratify=y)

# Initialize DecisionTreeClassifier
tree_clf = tree.DecisionTreeClassifier(random_state=123)

# Perform bias-variance decomposition
avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
    tree_clf, X_train.values, y_train.values, X_test.values, y_test.values, 
    loss='0-1_loss',
    random_seed=123)

print('Average expected loss: %.3f' % avg_expected_loss)
print('Average bias: %.3f' % avg_bias)
print('Average variance: %.3f' % avg_var)
```




    '\nhttps://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html\narrays\nsequence of indexables with same length / shape[0]\nAllowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.\n\ntest_size\nfloat or int, default=None\nIf float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25.\n\ntrain_size\nfloat or int, default=None\nIf float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split. If int, represents the absolute number of train samples. If None, the value is automatically set to the complement of the test size.\n\nrandom_state\nint, RandomState instance or None, default=None\nControls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls. See Glossary.\n\nshuffle\nbool, default=True\nWhether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.\n\nstratify\narray-like, default=None\nIf not None, data is split in a stratified fashion, using this as the class labels.\n'



    Average expected loss: 0.167
    Average bias: 0.160
    Average variance: 0.015



```python
#Exercise VIII: Statistical learning with DecisionTreeClassifier (cont)


X = filtered_df[['size', 'lead_time']]
y = filtered_df['booking']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, shuffle=False, stratify=None)

tree_clf = tree.DecisionTreeClassifier(random_state=123)

avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
    tree_clf, X_train.values, y_train.values, X_test.values, y_test.values, 
    loss='0-1_loss',
    random_seed=123)

print('Average expected loss: %.3f' % avg_expected_loss)
print('Average bias: %.3f' % avg_bias)
print('Average variance: %.3f' % avg_var)
```

    Average expected loss: 0.220
    Average bias: 0.213
    Average variance: 0.018


##### Exercise IX: Statistical learning with Bagging

X = filtered_df[['size', 'lead_time']]
y = filtered_df['booking']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, shuffle=True, stratify=y)

# Function to perform bias-variance decomposition with BaggingClassifier

def evaluate_bagging(n_estimators):
    # Initialize DecisionTreeClassifier
    tree_clf = DecisionTreeClassifier(random_state=123)

    # Initialize BaggingClassifier with DecisionTreeClassifier as base estimator
    bag = BaggingClassifier(estimator=tree_clf,
                            n_estimators=n_estimators,
                            random_state=123)

    # Perform bias-variance decomposition
    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
            bag, X_train.values, y_train.values, X_test.values, y_test.values, 
            loss='0-1_loss',
            random_seed=123)

    # Print the results
    print(f'n_estimators: {n_estimators}')
    print('Average expected loss: %.3f' % avg_expected_loss)
    print('Average bias: %.3f' % avg_bias)
    print('Average variance: %.3f' % avg_var)
    print('\n')

# Evaluate for different values of n_estimators

evaluate_bagging(100)
evaluate_bagging(500)
evaluate_bagging(1000)


