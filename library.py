import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import f1_score#, balanced_accuracy_score, precision_score, recall_score
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier

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
      X1 = titanic_features[self.column_list] # if no assert error happened here we are keeping the list of columns
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

