import numpy as np
import pandas as pd
import streamlit as st
from ahmedsabri import *
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from time import time

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


