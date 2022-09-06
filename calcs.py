import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

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

st.title('This simple app allows upload datasets, detects deviation and plot it if needed')
tab1=st.tabs('************************************************************************')
st.write('be sure that data is all numbers, no strings and the first raw is the column names')
st.header('first step is to upload your dataset')
data=st.file_uploader("Choose a CSV file")
if data is None:
    st.write('*Kindly upload valid data')
else:
    df = pd.read_csv(data)    
    st.write(df)
    st.header('2nd step is to choose how many standard deviation you need')
    a=st.slider('pick a number',2,10)
    st.write(a)
    st.write('**the columns that have data which has deviation more or less than**',a,'**is**')
    outlier_cols=outlier_columns(df,a).columns
    st.write(outlier_cols)
    click=st.button('plot deviated columns')
    if click==True:
        for column in outlier_cols:
            fig,ax=plt.subplots(figsize=(10,5))
            sns.lineplot(data=df,x=df.index,y=df[column],ax=ax)
            st.pyplot(fig)
    else:
        pass


