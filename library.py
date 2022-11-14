import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
# MAPPING TRANSFORMER
class MappingTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, mapping_column, mapping_dict:dict):
    assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
    self.mapping_dict = mapping_dict
    self.mapping_column = mapping_column  #column to focus on

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?
    
    #now check to see if all keys are contained in column
    column_set = set(X[self.mapping_column])
    keys_not_found = set(self.mapping_dict.keys()) - column_set
    if keys_not_found:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain these keys as values {keys_not_found}\n")

    #now check to see if some keys are absent
    keys_absent = column_set -  set(self.mapping_dict.keys())
    if keys_absent:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain keys for these values {keys_absent}\n")

    X_ = X.copy()
    X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
    #print(len(X_))
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
# OHE TRANSFORMER
class OHETransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, dummy_na=False, drop_first=False):  
    self.target_column = target_column  
    self.dummy_na = dummy_na
    self.drop_first = drop_first

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    # check for a valid database and see if there are any unknown columns
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.target_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.target_column}"'  

    # then just break them up as we did in class, using self.target_column as input
    X1 = pd.get_dummies(X, prefix=self.target_column,    
                           prefix_sep='_',   
                           columns=[self.target_column],
                           dummy_na=self.dummy_na,    
                           drop_first=self.drop_first)
    
    #print(len(X1))
    return X1

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
    

# DROP COLUMNS TRANSFORMER
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, column_list, action='drop'):
    assert action in ['keep', 'drop'], f'{self.__class__.__name__} action {action} not in ["keep", "drop"]'
    self.column_list = column_list
    self.action = action
  
  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    missing_columns = set(self.column_list) - set(X.columns.to_list()) # getting a list of missing columns
    
    X1 = X.copy() # copy the database before modifying
    if self.action == 'drop': # check if we are dropping columns
      if missing_columns: # if there is anything in the missing columns list
        print(f"\nWarning: {self.__class__.__name__} does not contain these columns to drop: {missing_columns}\n") # print a warning listing the missing cols
      X1 = X.drop(columns=self.column_list, errors = 'ignore')  # then drop them while ignoring errors
    elif self.action == 'keep': # if twe are keeping columns
      #check using assertion if the column list is a subset of the database column list
      assert set(self.column_list) <= set(X.columns.to_list()), f'{self.__class__.__name__} does not contain these columns to keep: "{missing_columns}"' 
      X1 = X1[self.column_list] # if no assert error happened here we are keeping the list of columns
    else: # and this is the "something went wrong" option that just exits out of the program
      print('Something went terribly, terribly wrong!')
      exit()
    
    #print(len(X1))
    return X1

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
# PEARSON TRANSFORMER
class PearsonTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, threshold):
    assert (type(threshold) == int or type(threshold) == float), f'{self.__class__.__name__} constructor expected a number but got {type(threshold)} instead.'
    self.threshold = threshold   
  
  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, df):
    assert isinstance(df, pd.core.frame.DataFrame), f'{self.__class__.__name__} constructor expected a Dataframe but got {type(df)} instead.'
    
    df_corr = transformed_df.corr(method='pearson')
    masked_df = df_corr.abs() >= self.threshold
    upper_mask = np.triu(masked_df, 1)
    column_ind = set(np.where(upper_mask)[1]) 
    correlated_columns = [masked_df.columns.values[j] for i, j in enumerate(column_ind)]
    new_df = transformed_df.drop(columns=correlated_columns)
    
    #print(len(new_df))
    return new_df

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
# SIGMA TRANSFORMER
class Sigma3Transformer(BaseEstimator, TransformerMixin):
  def __init__(self, column_name):
    self.column_name = column_name

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, df):
    assert isinstance(df, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(df)} instead.'
    assert self.column_name in df.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.column_name}"'  
    assert all([isinstance(v, (int, float)) for v in df[self.column_name].to_list()])

    df_mean = df[self.column_name].mean()
    s3max = df_mean + 3*df[self.column_name].std()
    s3min = df_mean - 3*df[self.column_name].std()
   
    df1 = df.copy()
    df1[self.column_name] = df[self.column_name].clip(lower=s3min, upper=s3max)
    
    #print(len(df1))
    return df1

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
# TUKEY TRANSFORMER
class TukeyTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, fence):
    self.target_column = target_column
    self.fence = fence

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, df):
    assert isinstance(df, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(df)} instead.'
    assert self.target_column in df.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.target_column}"'  
    assert self.fence in ('inner', 'outer'), f'{self.__class__.__name__}.transform unknown fence condition"{self.fence}"'
    assert all([isinstance(v, (int, float)) for v in df[self.target_column].to_list()])

    q1 = df[self.target_column].quantile(0.25)
    q3 = df[self.target_column].quantile(0.75)
    iqr = q3-q1
    if self.fence == 'inner':
      k = 1.5
    elif self.fence == 'outer':
      k = 3
    else:
      print('Something went horribly wrong! Exiting...')
      exit()
    
    low = q1 - k * iqr
    high = q3 + k * iqr  

    df1 = df.copy()
    df1[self.target_column] = df[self.target_column].clip(lower=low, upper=high)
    
    #print(len(df1))
    return df1

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

# MINMAX TRANSFORMER  
class MinMaxTransformer(BaseEstimator, TransformerMixin):
  from sklearn.preprocessing import MinMaxScaler

  def __init__(self):
    pass  #takes no arguments

  #fill in rest below
  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    
    print(len(X))
    return X

  def transform(self, df):
    assert isinstance(df, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(df)} instead.'
    
    new_df = df.copy() # copy the df
    col_names = new_df.columns.tolist() # copy the column names
    scaler = self.MinMaxScaler() # run the scaler from internal import
    numpy_result = scaler.fit_transform(new_df) # do the transform  
    new_df = pd.DataFrame(numpy_result) # turn the result back into a dataframe
    new_df.columns = col_names # restore the column names
    new_df.describe(include='all').T
    
    #print(len(new_df))
    return new_df
  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

  
# KNN TRANSFORMER
class KNNTransformer(BaseEstimator, TransformerMixin):
  from sklearn.impute import KNNImputer

  def __init__(self,n_neighbors=5, weights="uniform"):
    assert (type(n_neighbors) == int and n_neighbors > 0), f'{self.__class__.__name__} constructor expected a positive integer but got {n_neighbors} instead.'
    assert weights in ['uniform', 'distance'], f'{self.__class__.__name__} action {weights} not in ["uniform", "distance"]'
    self.n_neighbors = n_neighbors   
    self.weights = weights

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, df):
    assert isinstance(df, pd.core.frame.DataFrame), f'{self.__class__.__name__} constructor expected a Dataframe but got {type(df)} instead.'
    
    imputer = self.KNNImputer(n_neighbors = self.n_neighbors,    
                              weights = self.weights,    
                              add_indicator = False) 
    
    col_names = df.columns.tolist()
    new_df = pd.DataFrame(imputer.fit_transform(df))
    new_df.columns = col_names
    
    #print(len(new_df))
    return new_df

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
# FIND RANDOM STATE FUNCTION 
def find_random_state(features_df, labels, n=200):
  assert isinstance(features_df, pd.core.frame.DataFrame), f'expected a Dataframe but got {type(features_df)} instead.'
  # assert isinstance(labels, list), f'expected a list but got {type(labels)} instead.'
  assert (type(n) == int and n > 0), f'expected n to be a positive integer, got a {n} instead.'
  
  var = []  
  for i in range(1, n):
    train_X, test_X, train_y, test_y = train_test_split(features_df,
                                                        labels,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=i,
                                                        stratify=labels)
    model.fit(train_X, train_y)                #train model
    train_pred = model.predict(train_X)        #predict against training set
    test_pred = model.predict(test_X)          #predict against test set
    train_f1 = f1_score(train_y, train_pred)   #F1 on training predictions
    test_f1 = f1_score(test_y, test_pred)      #F1 on test predictions
    f1_ratio = test_f1/train_f1                #take the ratio
    var.append(f1_ratio)

  rs_value = sum(var)/len(var)                   #get average ratio value
  idx = np.array(abs(var - rs_value)).argmin()   #find the index of the smallest value
  return idx

# DATASET SETUP FUNCTION (GENERIC)
def dataset_setup(full_table, label_column_name:str, the_transformer, rs, ts=.2):
  from sklearn.model_selection import train_test_split
  
  assert isinstance(full_table, pd.core.frame.DataFrame), f'Expected a Dataframe but got {type(full_table)} instead.'
  assert label_column_name in full_table.columns.to_list(), f'Unknown column: "{label_column_name}"'
  assert isinstance(rs, int), f'The rs value has to be a positive int, you provided: {rs}'
  assert isinstance(ts, (int, float)), f'The ts value has to be a positive int or float, you provided: {ts}'

  full_table_features = full_table.drop(columns=label_column_name)
  labels = full_table[label_column_name].to_list()

  X_train, X_test, y_train, y_test = train_test_split(full_table_features,
                                                      labels,
                                                      test_size=ts,
                                                      shuffle=True,
                                                      random_state=rs,
                                                      stratify=labels)
  
  X_train_transformed = the_transformer.fit_transform(X_train)
  X_test_transformed = the_transformer.fit_transform(X_test)
  
  x_trained_numpy = X_train_transformed.to_numpy()
  x_test_numpy = X_test_transformed.to_numpy()
  y_train_numpy = np.array(y_train)
  y_test_numpy = np.array(y_test)

  return x_trained_numpy, x_test_numpy, y_train_numpy, y_test_numpy


# TITANIC TABLE SPECIFIC TRANSFORMER PIPELINE
titanic_transformer = Pipeline(steps=[
  ('drop', DropColumnsTransformer(['Age', 'Gender', 'Class', 'Joined', 'Married',  'Fare'], 'keep')),
  ('gender', MappingTransformer('Gender', {'Male': 0, 'Female': 1})),
  ('class', MappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
  ('ohe', OHETransformer(target_column='Joined')),
  ('age', TukeyTransformer(target_column='Age', fence='outer')), #from chapter 4
  ('fare', TukeyTransformer(target_column='Fare', fence='outer')), #from chapter 4
  ('minmax', MinMaxTransformer()),  #from chapter 5
  ('imputer', KNNTransformer())  #from chapter 6
  ], verbose=True)

# TITANIC TABLE SPECIFIC SETUP FUNCTION
def titanic_setup(titanic_table, transformer=titanic_transformer, rs=40, ts=.2):
  x_trained_numpy, x_test_numpy, y_train_numpy,  y_test_numpy = dataset_setup(titanic_table, 'Survived',
                                                                              transformer,
                                                                              rs, 
                                                                              ts)
  return x_trained_numpy, x_test_numpy, y_train_numpy,  y_test_numpy

# CUSTOMER TABLE SPECIFIC TRANSFORMER PIPELINE
customer_transformer = Pipeline(steps=[
  ('id', DropColumnsTransformer(column_list=['ID'])),
  ('os', OHETransformer(target_column='OS')),
  ('isp', OHETransformer(target_column='ISP')),
  ('level', MappingTransformer('Experience Level', {'low': 0, 'medium': 1, 'high':2})),
  ('gender', MappingTransformer('Gender', {'Male': 0, 'Female': 1})),
  ('time spent', TukeyTransformer('Time Spent', 'inner')),
  ('minmax', MinMaxTransformer()),
  ('imputer', KNNTransformer())
  ], verbose=True)

# CUSTOMER TABLE SPECIFIC SETUP FUNCTION
def customer_setup(customer_table, transformer=customer_transformer, rs=76, ts=.2):
  x_trained_numpy, x_test_numpy, y_train_numpy,  y_test_numpy = dataset_setup(customer_table, 'Rating',
                                                                              transformer,
                                                                              rs, 
                                                                              ts)
  return x_trained_numpy, x_test_numpy, y_train_numpy,  y_test_numpy

# THRESHOLD RESULTS FUNCTION
def threshold_results(thresh_list, actuals, predicted):
  result_df = pd.DataFrame(columns=['threshold', 'precision', 'recall', 'f1', 'accuracy'])
  for t in thresh_list:
    yhat = [1 if v >=t else 0 for v in predicted]
    #note: where TP=0, the Precision and Recall both become 0
    precision = precision_score(actuals, yhat, zero_division=0)
    recall = recall_score(actuals, yhat, zero_division=0)
    f1 = f1_score(actuals, yhat)
    accuracy = accuracy_score(actuals, yhat)
    result_df.loc[len(result_df)] = {'threshold':t, 'precision':precision, 'recall':recall, 'f1':f1, 'accuracy':accuracy}

  result_df = result_df.round(2)

  #Next bit fancies up table for printing. See https://betterdatascience.com/style-pandas-dataframes/
  #Note that fancy_df is not really a dataframe. More like a printable object.
  headers = {
    "selector": "th:not(.index_name)",
    "props": "background-color: #800000; color: white; text-align: center"
  }
  properties = {"border": "1px solid black", "width": "65px", "text-align": "center"}

  fancy_df = result_df.style.format(precision=2).set_properties(**properties).set_table_styles([headers])
  return (result_df, fancy_df)

# HALVING SEARCH FUNCTION
def halving_search(model, grid, x_train, y_train, factor=3, scoring='roc_auc'):
  halving_cv = HalvingGridSearchCV(model,
                                   grid, 
                                   scoring = scoring, 
                                   n_jobs=-1,
                                   min_resources="exhaust",
                                   factor = factor, 
                                   cv=5,
                                   random_state=1234,
                                   refit=True,  #remembers the best combo and gives us back that model already trained and ready for testing
                                   )

  grid_result = halving_cv.fit(x_train, y_train)
  return grid_result

