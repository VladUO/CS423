import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

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

    return X1

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result