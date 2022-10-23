import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score#, balanced_accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# MAPPING TRANSFORMER
class MappingTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, mapping_column, mapping_dict:dict):
    assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
    self.mapping_dict = mapping_dict
    self.mapping_column = mapping_column  #column to focus on

  def fit(self, df1, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X1

  def transform(self, df1):
    assert isinstance(df1, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(df1)} instead.'
    assert self.mapping_column in df1.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?
    
    #now check to see if all keys are contained in column
    column_set = set(df1[self.mapping_column])
    keys_not_found = set(self.mapping_dict.keys()) - column_set
    if keys_not_found:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain these keys as values {keys_not_found}\n")

    #now check to see if some keys are absent
    keys_absent = column_set -  set(self.mapping_dict.keys())
    if keys_absent:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain keys for these values {keys_absent}\n")

    df1_copy = df1.copy()
    df1_copy[self.mapping_column].replace(self.mapping_dict, inplace=True)
    #print(len(df1_copy))
    return df1_copy

  def fit_transform(self, df1, y = None):
    result = self.transform(df1)
    return result
  
# OHE TRANSFORMER
class OHETransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, dummy_na=False, drop_first=False):  
    self.target_column = target_column  
    self.dummy_na = dummy_na
    self.drop_first = drop_first

  def fit(self, df2, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return df2

  def transform(self, df2):
    # check for a valid database and see if there are any unknown columns
    assert isinstance(df2, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(df2)} instead.'
    assert self.target_column in df2.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.target_column}"'  

    # then just break them up as we did in class, using self.target_column as input
    df2_copy = pd.get_dummies(df2, prefix=self.target_column,    
                           prefix_sep='_',   
                           columns=[self.target_column],
                           dummy_na=self.dummy_na,    
                           drop_first=self.drop_first)
    
    #print(len(df2_copy))
    return df2_copy

  def fit_transform(self, df2, y = None):
    result = self.transform(df2)
    return result
    

# DROP COLUMNS TRANSFORMER
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, column_list, action='drop'):
    assert action in ['keep', 'drop'], f'{self.__class__.__name__} action {action} not in ["keep", "drop"]'
    self.column_list = column_list
    self.action = action
  
  def fit(self, df3, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return df3

  def transform(self, df3):
    assert isinstance(df3, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(df3)} instead.'
    missing_columns = set(self.column_list) - set(df3.columns.to_list()) # getting a list of missing columns
    
    df3_copy = df3.copy() # copy the database before modifying
    if self.action == 'drop': # check if we are dropping columns
      if missing_columns: # if there is anything in the missing columns list
        print(f"\nWarning: {self.__class__.__name__} does not contain these columns to drop: {missing_columns}\n") # print a warning listing the missing cols
      df3_copy = df3.drop(columns=self.column_list, errors = 'ignore')  # then drop them while ignoring errors
    elif self.action == 'keep': # if twe are keeping columns
      #check using assertion if the column list is a subset of the database column list
      assert set(self.column_list) <= set(df.columns.to_list()), f'{self.__class__.__name__} does not contain these columns to keep: "{missing_columns}"' 
      df3_copy = titanic_features[self.column_list] # if no assert error happened here we are keeping the list of columns
    else: # and this is the "something went wrong" option that just exits out of the program
      print('Something went terribly, terribly wrong!')
      exit()
    
    #print(len(df3_copy))
    return df3_copy

  def fit_transform(self, df3, y = None):
    result = self.transform(df3)
    return result
  
# PEARSON TRANSFORMER
class PearsonTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, threshold):
    assert (type(threshold) == int or type(threshold) == float), f'{self.__class__.__name__} constructor expected a number but got {type(threshold)} instead.'
    self.threshold = threshold   
  
  def fit(self, df4, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, df4):
    assert isinstance(df4, pd.core.frame.DataFrame), f'{self.__class__.__name__} constructor expected a Dataframe but got {type(df4)} instead.'
    
    df_corr = transformed_df.corr(method='pearson')
    masked_df = df_corr.abs() >= self.threshold
    upper_mask = np.triu(masked_df, 1)
    column_ind = set(np.where(upper_mask)[1]) 
    correlated_columns = [masked_df.columns.values[j] for i, j in enumerate(column_ind)]
    new_df = transformed_df.drop(columns=correlated_columns)
    
    #print(len(df4_copy))
    return new_df

  def fit_transform(self, df4, y = None):
    result = self.transform(df4)
    return result
  
# SIGMA TRANSFORMER
class Sigma3Transformer(BaseEstimator, TransformerMixin):
  def __init__(self, column_name):
    self.column_name = column_name

  def fit(self, df5, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, df5):
    assert isinstance(df5, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(df5)} instead.'
    assert self.column_name in df5.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.column_name}"'  
    assert all([isinstance(v, (int, float)) for v in df5[self.column_name].to_list()])

    df_mean = df5[self.column_name].mean()
    s3max = df_mean + 3*df5[self.column_name].std()
    s3min = df_mean - 3*df5[self.column_name].std()
   
    df5_copy = df5.copy()
    df5_copy[self.column_name] = df5[self.column_name].clip(lower=s3min, upper=s3max)
    
    #print(len(df5_copy))
    return df5_copy

  def fit_transform(self, df5, y = None):
    result = self.transform(df5)
    return result
  
# TUKEY TRANSFORMER
class TukeyTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, fence):
    self.target_column = target_column
    self.fence = fence

  def fit(self, df6, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return df6

  def transform(self, df6):
    assert isinstance(df6, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(df6)} instead.'
    assert self.target_column in df6.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.target_column}"'  
    assert self.fence in ('inner', 'outer'), f'{self.__class__.__name__}.transform unknown fence condition"{self.fence}"'
    assert all([isinstance(v, (int, float)) for v in df6[self.target_column].to_list()])

    q1 = df6[self.target_column].quantile(0.25)
    q3 = df6[self.target_column].quantile(0.75)
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

    df6_copy = df6.copy()
    df6_copy[self.target_column] = df[self.target_column].clip(lower=low, upper=high)
    
    #print(len(df6_copy))
    return df6_copy

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

# MINMAX TRANSFORMER  
class MinMaxTransformer(BaseEstimator, TransformerMixin):
  from sklearn.preprocessing import MinMaxScaler

  def __init__(self):
    pass  #takes no arguments

  #fill in rest below
  def fit(self, df7, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    
    return df7

  def transform(self, df7):
    assert isinstance(df7, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(df7)} instead.'
    
    df7_copy = df7.copy() # copy the df
    col_names = df7_copy.columns.tolist() # copy the column names
    scaler = self.MinMaxScaler() # run the scaler from internal import
    numpy_result = scaler.fit_transform(df7_copy) # do the transform  
    df7_copy = pd.DataFrame(numpy_result) # turn the result back into a dataframe
    df7_copy.columns = col_names # restore the column names
    df7_copy.describe(include='all').T
    
    #print(len(new_df))
    return df7_copy
  
  def fit_transform(self, df7, y = None):
    result = self.transform(df7)
    return result

  
# KNN TRANSFORMER
class KNNTransformer(BaseEstimator, TransformerMixin):
  from sklearn.impute import KNNImputer

  def __init__(self,n_neighbors=5, weights="uniform"):
    assert (type(n_neighbors) == int and n_neighbors > 0), f'{self.__class__.__name__} constructor expected a positive integer but got {n_neighbors} instead.'
    assert weights in ['uniform', 'distance'], f'{self.__class__.__name__} action {weights} not in ["uniform", "distance"]'
    self.n_neighbors = n_neighbors   
    self.weights = weights

  def fit(self, df8, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, df8):
    assert isinstance(df8, pd.core.frame.DataFrame), f'{self.__class__.__name__} constructor expected a Dataframe but got {type(df8)} instead.'
    
    imputer = self.KNNImputer(n_neighbors = self.n_neighbors,    
                              weights = self.weights,    
                              add_indicator = False) 
    
    col_names = df8.columns.tolist()
    df8_copy = pd.DataFrame(imputer.fit_transform(df8))
    df8_copy.columns = col_names
    
    #print(len(df8_copy))
    return df8_copy

  def fit_transform(self, df8, y = None):
    result = self.transform(df8)
    return result

