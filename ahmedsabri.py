import numpy as np
import pandas as pd
import scipy.stats as stats
import os
import matplotlib.pyplot as plt
from datetime import datetime,date, timedelta
import shutil
from time import time
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.base import clone



def prepare(data,transpose=False):
    '''
    The function returns a data frame with column lablels thate are combined strings from the first couple of raws

    - inputs  1st argument: dataframe, 2nd argument, transpose=True, if the first thing to do is to transpose the data
    - syntax, prepare(dataframe,transpose=False)
    - returns : dataframe
    '''
    df=data.copy()
    if transpose==True:
        df=df.transpose()
    else:
        pass
    df.dropna(how='all',inplace=True)
    df.dropna(how='all',axis=1,inplace=True)
    df.reset_index(drop=True,inplace=True)
    return df

def initiate_headers(data):
    '''
    The function concacenate the first couple of rows which usually are text and make those text as columns header,
    It is important to check that all the rows above the numbers are text for this function to act correctly,
    if not all first rows are texe you can drop some rows and then reindex before applying this function
    
    - Inputs : Dataframe with headers names as numbers
    - Returns : Dataframe with headers name are the first raw of text concacenated
    '''
    df=data.copy()
    middle_column= df.columns[df.shape[1]//2]
    x=df.loc[~(df[middle_column].str.contains('[a-zA-Z]',na=False,regex=True)),middle_column].index[0]
    df[0:x-1]=df[0:x-1].values.astype(str)
    df.columns=df.iloc[0:x-1].fillna('').astype(str).apply(' '.join).str.strip()
    df.columns=df.columns.str.replace(" ","_")
    df.columns=df.columns.str.lower()
    df.columns=df.columns.str.replace("-","_")
    df=df.iloc[x:,:].reset_index(drop=True)
    return df    

def convert_string_to_nan(data):
    '''
    This function converts any string to nan, columnwise
    
    -inputs : dataframe which columns contains number and strings
    - output : dataframe which columns contains numbers and nans
    '''
    df=data.copy()
    for column in df.columns:
        try:
            df.loc[df[column].str.contains("[a-zA-Z]",na=False,regex=True),column]=np.nan
        except:
            print(f'{column} can not be processed')
    return df

def to_float(data,x=0):
    '''
    converting all columns to float starting from x column, where x is the position of the columns
    Nat values doesn't allow the column to be converted to float, be sure to remove all NAT values
    
    - inputs :  dataframe and the first column position to start converting from
      syntax to_float(data,x=0)
      
    - output : dataframe that all of its columns are float, if possible
    '''
    df=data.copy()
    columns = df.columns[x:]
    for column in columns:
        try:
            df[column]=df[column].astype(float)
        except:
            pass
    return df

def outlier_columns(df,a=4):
    '''
    This function call columns which contains any outliers larger than a value ( default a=4), a represents standard deviation
    
    - inputs : dataframe and a ( number of standard deviations)
    - syntax : outlier_columns(df,a=4)
    - output : columns that have the outliers > a* standard deviation of that column
    '''    
    z_scores = stats.zscore(df[df.describe().columns],nan_policy='omit')
    z_scores.fillna(0,inplace=True)
    abs_z_scores = np.abs(z_scores)
    (abs_z_scores>a).any(axis=0)
    outliers_columns=abs_z_scores.columns[(abs_z_scores>a).any(axis=0)]
    return df[outliers_columns]

def cell_repetion(data,row_num=0):
    '''
    This is usually done before initiating headers name, where you can rename cells as its previous cell in the same row
    (default row number is zero, but it can be done to any row)
    
    input: dataframe and row_number (default zero, i.e top row)
    Example-input: h2o,nan,nan,nan,hcl,nan,nan,nan
    Example-output: h2o,h2o,h2o,h2o,hcl,hcl,hcl,hcl
    '''
    df=data.copy()
    for i in df.columns:
        try:
            if type(df.loc[0,i+1])!=str:
                df.loc[0,i+1]=df.loc[0,i]
        except:
            pass
    return df

def keep_columns(data,threshold=0.99):
    '''
    This function keep the column which contains non na values above the threshold value (default 99 %) and remove all other columns
    - input : dataframe, threshold value
    - syntax: keep_data(data,threshold=0.99)
    -output : dataframe with all columns contaiing nan values more than threshold values are removed
    
    EXAMPLE : i want to remove the column that has 2/3 of its values as nan : Keep_data(data,threshold=0.667)
    '''
    df=data.copy()
    df.dropna(axis='columns', how='any', thresh=int(df.shape[0]*threshold), inplace=True)
    return df

def keep_rows(data,threshold=0.99,indexreset=False):
    '''
    This function keep the rows which contains non na values above the threshold value (default 99 %) and remove all other rows,
    additionally it is optional to reset index or not ( default indexreset=False)
    - input : dataframe, threshold value and indexreset(default False)
    - syntax: keep_data(data,threshold=0.99,indexreset=False)
    -output : dataframe with all raw containing nan values more than threshold values are removed
    '''
    df=data.copy()
    df.dropna(axis=0, how='any', thresh=int(df.shape[1]*threshold), inplace=True)
    if indexreset==True:
        df.reset_index(drop=True,inplace=True)
    return df

def plotting(df):
    for column in df.columns:
        try:
            df[column].plot()
            plt.xlabel('hrs')
            plt.ylabel(column)
            plt.show()
        except:
            print(f'{column} can not be plotted')
        
def remove_spot(data):
    '''
    This function removes spot reading column from raw dataframe before transpose
    - input : dataframe
    - syntax: remove_spot(data)
    -output : dataframe with spot reading column removed
    '''
    df=data.copy()
    for column in df.columns:
        try:
            if (type(df.iloc[10,df.columns.get_loc(column)])!= str) and (type(df.iloc[10,df.columns.get_loc(column)+1])== str) and (type(df.iloc[10,df.columns.get_loc(column)-1])== str) :
                x=column
                df.drop(x,axis=1,inplace=True)
        except:
            pass
    return df

def remove_duplicated_columns(data, same_name_only=False):
    '''
    This function removes duplicated columns
    
    - parameters: same_name_only: if true, means that duplicated columns
    will be removed only if the have the same column name 
    
    if false, means that duplicated columns
    will be removed even if they do not have the same column name
    
    -syntax: remove_duplicated_columns(dataframe, same_name_only=False)
    
    -output: dataframe without duplicated columns
  
    '''
    df=data.copy()
    if same_name_only == False:
        return df.T.drop_duplicates().T
    else:
        return df.loc[:,~df.columns.duplicated()]

X=[0,1,2] 
y=[3]       
    
        
def estimators_repeater(estimators=[RandomForestClassifier(),AdaBoostClassifier(),SVC()],tr_slicer=(None,None),tst_slicer=(None,None),loops=500,scorer=accuracy_score,X=X,y=y):
    '''This function aims to train list of supplied estimators with selcted slices of datasets for as many time as required(default 500)
    and then produce a list of training score, test score and time used for each estimator
    
    - It is important to import all used estimators, score to be used
    inputs :
    - estimators : a list of estimators, deafult is Randomforest, Adaboost and support vector machine
    
    - tr_slicer : slicer for the number of observations needed in the training, default is all samples [:], slicer should be tuples
    of integers(starter,ender) default is(None,None)
    
    - tst_slicer : slicer for the number of samples to be tested at, default is all samples [0:-1], slicer should be tuples
    of integers(starter,ender) default is(0,-1)
    
    -loops : int, is a number of loops needed : default 500
    
    -scorer : default is accuracy_score, but it can be anything choosen from sklearn.metrics but if it is something calculated by
    methods other than accuracy, it should be modified in the 
    
    -X= features in the form of dataframe or np.array of 2 dimensions
    -y= target in the form of dataframe, series or np.array
    
    Returns : 3 global dataframes for training score(training_score_df),
    testing score(testing_score_df) and time (time_df) used for each estimator for fitting and predicting
    
    
    Example, fitting first 200 samples for SVC() and RandomForestClassifier() for 400 loop and scorer is accuracy for full test dataset:
    
    estimators_repeater(estimators=[RandomForestClassifier(),SVC()],tr_slicer=(0,200),loops=400,scorer=accuracy_score,X=X,y=y)
    
    '''
    
    training_score={}
    testing_score={}
    timing={}
    for clf in estimators:
        clf_name = clf.__class__.__name__
        training_score[clf_name]=[]
        testing_score[clf_name]=[]
        timing[clf_name]=[]
       
    for i in range (loops):
        k1=time()
        X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=i)
        for clf in estimators:
            a=time()
            clf_name = clf.__class__.__name__
            clean_clf=clone(clf)
            clean_clf.fit(X_train[tr_slicer[0]:tr_slicer[1]],y_train[tr_slicer[0]:tr_slicer[1]])
            training_score[clf_name].append(scorer(y_train[tr_slicer[0]:tr_slicer[1]],clean_clf.predict(X_train[tr_slicer[0]:tr_slicer[1]])))
            testing_score[clf_name].append(scorer(y_test[tst_slicer[0]:tst_slicer[1]],clean_clf.predict(X_test[tst_slicer[0]:tst_slicer[1]])))
            b=time()
            timing[clf_name].append(b-a)
        k2=time()
        print(f'loop number {i} out of {loops} took {k2-k1} seconds')
    
    global training_score_df
    training_score_df=pd.DataFrame(training_score)
    global testing_score_df
    testing_score_df=pd.DataFrame(testing_score)
    global timing_df
    timing_df=pd.DataFrame(timing)    
        
