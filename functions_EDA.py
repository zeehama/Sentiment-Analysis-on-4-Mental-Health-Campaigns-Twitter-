#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.offline import iplot

import warnings
warnings.filterwarnings('ignore')

#********************************************************************************************************************

def get_info_dataset(data, bool = True):
    """ 
    Function to give a broad overview of the dataset
    input: dataframe, bool
        bool = True (default) prints info
        bool = False only returns num and cat columns
    prints info, head, tail, describe, num value columns, and cat value columns
    returns num_var & cat_var
    """
    
    num_var = data.select_dtypes(include=['int64', 'float64']).columns
    categ_var = data.select_dtypes(include=['category', 'object']).columns
    
    if bool == True:
    
        print('Basic information\n','---------------------------------------------')
        data.info()

        print('\nFirst 5 rows\n','---------------------------------------------')
        print(data.head())

        print('\nLast 5 rows\n','---------------------------------------------')
        print(data.tail())

        print('\nBasic statistics\n','---------------------------------------------')
        print(data.describe().T)

        print('\nNumerical variables are:\n', num_var)
        print('-------------------------------------------------')

        print('Categorical variables are:\n', categ_var)
        print('-------------------------------------------------') 
    
    return num_var,categ_var

#********************************************************************************************************************

def percentage_nullValues(data):
    """
    Function that calculates the percentage of missing values in every column of your dataset
    input: data --> dataframe
    """
    null_perc = round(data.isnull().sum() / data.shape[0],3) * 100.00
    null_perc = pd.DataFrame(null_perc, columns=['Percentage_NaN'])
    null_perc= null_perc.sort_values(by = ['Percentage_NaN'], ascending = False)
    
    return null_perc

#********************************************************************************************************************

def check_dup(df):
    '''
    Function to give the shape of the original df
    and the shape of the df with all duplicates dropped
    if they are not the same then there are duplicates
    '''
    
    #create a backup of our dataset (df)
    df_copy = df.copy()
    
    df_c = df.drop_duplicates()
    ## Always check everyhting!
    print('Shape of the raw data', df_copy.shape)
    print('........................................')
    print('Shape of the new data', df_c.shape)

#********************************************************************************************************************

def select_threshold(data, thr):
    """
    Function that  \calculates the percentage of missing values in every column of your 
    dataset and drops those above a threshold
    input: data --> dataframe
    
    """
    null_perc = percentage_nullValues(data)
      
    col_keep = null_perc[null_perc['Percentage_NaN'] < thr]
    col_drop = null_perc[null_perc['Percentage_NaN'] > thr]
    col_keep = list(col_keep.index)
    col_drop = list(col_drop.index)
    print('Columns to keep:',len(col_keep))
    print('Those columns have a percentage of NaN less than', 
          str(thr), ':')
    print()
    print(col_keep)
    
    print('*********************************************************')
    print('Columns to drop:',len(col_drop))
    print('Those columns have a percentage of NaN more than', 
          str(thr), ':')
    print()
    print(col_drop)
    
    data_c= data[col_keep]
    
    return data_c

#********************************************************************************************************************


def fill_na(data):
    """
    Function to fill NaN with mode (categorical variables) and mean (numerical variables)
    input: data -> df
    """
    for column in data:
        if data[column].dtype != 'object':
            data[column] = data[column].fillna(data[column].mean())  
        else:
            data[column] = data[column].fillna(data[column].mode()[0]) 
    
    print('Number of missing values on your dataset are')
    print()
    print(data.isnull().sum())
    return data

#********************************************************************************************************************

def corrCoef(data):
    """
    Function aimed to calculate the corrCoef between each pair of variables
   
    input: data->dataframe        
    """
    num_var, categ_var = get_info_dataset(data, True)
    data_num = data[num_var]
    data_corr = data_num.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(data_corr,
                xticklabels = data_corr.columns.values,
               yticklabels = data_corr.columns.values,
               annot = True, vmax=1, vmin=-1, center=0, cmap= sns.color_palette("RdBu_r", 7))

#********************************************************************************************************************

def corrCoef_Threshold(data, threshold):
    """
    Function aimed to calculate the corrCoef between each pair of variables
    
    input: data->dataframe 
           threshold -> True: we want to keep the variables with a corrCoef higher than the income
                        False: we want to keep all values, no filtering
            
            """
    num_var, categ_var = get_info_dataset(data, True)
    data_num = data[num_var]
    data_corr = data_num.corr()
    data_cols = data_corr.columns
    
    if threshold == True:
        data_corr= pd.DataFrame(data_corr.unstack().sort_values(ascending = False),
                                columns = ['corrCoef'])
       # threshold that I want to select. I will keep the variables with a corrCoef higher than the threshols
        thr = float(input('Threshold? (in positive sign, please) '))
        data_corr = data_corr[(data_corr.corrCoef >thr)| (data_corr.corrCoef< -thr)].unstack()

        data_corr = pd.DataFrame(data_corr)

        # Create the plot
        plt.figure(figsize=(10,8))
        sns.heatmap(data_corr,xticklabels = data_cols,
                    yticklabels = data_cols,
                            annot = True, vmax=1, vmin=-1, center=0, cmap= sns.color_palette("RdBu_r", 7))
        
    else:
        plt.figure(figsize=(10,8))
        sns.heatmap(data_corr,
                    xticklabels = data_corr.columns.values,
                   yticklabels = data_corr.columns.values,
                   annot = True, vmax=1, vmin=-1, center=0, cmap= sns.color_palette("RdBu_r", 7))
    
    return data_corr

#********************************************************************************************************************

def outlier_treatment(df, colname):
    """
    Function that drops the Outliers based on the IQR upper and lower boundaries
    input: df --> dataframe
           colname --> str, name of the column

    """

    # Calculate the percentiles and the IQR
    Q1,Q3 = np.percentile(df[colname], [25,75])
    IQR = Q3 - Q1
    
    # Calculate the upper and lower limit
    lower_limit = Q1 - (1.5 * IQR)
    upper_limit = Q3 + (1.5 * IQR)
    
    # Drop the suspected outliers
    df_clean = df[(df[colname] > lower_limit) & (df[colname] < upper_limit)]
    
    print('Shape of the raw data:', df.shape)
    print('..................')
    print('Shape of the cleaned data:', df_clean.shape)
    return df_clean

#********************************************************************************************************************

def OutLiersBox(df,nameOfFeature):
    """
    Function to create a BoxPlot and visualise:
    - All Points in the Variable
    - Suspected Outliers in the variable

    """
    trace0 = go.Box(
        y = df[nameOfFeature],
        name = "All Points",
        jitter = 0.3,
        pointpos = -1.8,
        boxpoints = 'all', #define that we want to plot all points
        marker = dict(
            color = 'rgb(7,40,89)'),
        line = dict(
            color = 'rgb(7,40,89)')
    )

    trace1 = go.Box(
        y = df[nameOfFeature],
        name = "Suspected Outliers",
        boxpoints = 'suspectedoutliers', # define the suspected Outliers
        marker = dict(
            color = 'rgb(8,81,156)',
            outliercolor = 'rgba(219, 64, 82, 0.6)',
            line = dict(
                outliercolor = 'rgba(219, 64, 82, 0.6)',
                outlierwidth = 2)),
        line = dict(
            color = 'rgb(8,81,156)')
    )

    data = [trace0,trace1]

    layout = go.Layout(
        title = "{} Outliers".format(nameOfFeature)
    )

    fig = go.Figure(data=data,layout=layout)
    fig.show()   
    
#********************************************************************************************************************
    
def half_corr_matrix(df):
    '''Function to create the bottom left triangle of a correlation matrix
    import - dataframe only'''
    
    plt.figure(figsize = (15, 7))
    corr_matrix=df.corr()
    
    #create a mask 
    mask = np.zeros_like(corr_matrix)
    mask[np.triu_indices_from(mask)] = True
    
    # Draw the heatmap with the mask
    ax = sns.heatmap(corr_matrix, mask=mask, annot=True, linewidths=.75, cmap ="Blues")
    plt.show()

#********************************************************************************************************************
    
def create_pyplot_fig(Dataframe, var1, figname):
    """Function to create and show plots:
       #inputs: Dataframe (dataframe that we want to visualise)
            var1 ('str' with the variable that we want to plot)
            figname ('str' with the fig name once we save it)"""
    
    from matplotlib.pyplot import figure
    figure(num=None, figsize=(20, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.style.use('seaborn-white')  
    x = np.arange(0,len(Dataframe),1)
    ax = plt.subplot(111)
    ax.plot(x, Dataframe[var1], label = var1) #label will be our Variable 1!
          
    leg = ax.legend()
    ax.yaxis.set_label_text(str(figname))
    plt.axhline(y=0, color='k', linestyle='--') #We create a line with y= 0 and linestyle = "--"
    plt.autoscale(enable=True, axis='x', tight=True) #We auto-adjust the scale
    plt.savefig(str(figname))
    plt.show();    
    
#********************************************************************************************************************    
    
def all_hist(df):
    '''Function to create histograms for every numeric column in dataset
       a sentence will print out to say which columns can't be used'''
       
    for col in df.columns:
        try:      
            df.hist(column=col)
        except ValueError:
            print(col, 'can not be represented as a histogram')

#********************************************************************************************************************
            
def all_box(df):
    '''Function to create boxplots for every numeric column in dataset
       a sentence will print out to say which columns can't be used'''
    
    for col in df.columns:
        try:
            plt.figure()
            df.boxplot([col])
        except ValueError:
            print(col, 'can not be represented as a boxplot')
            
#********************************************************************************************************************
            
def all_unique(df):
    '''Function to print unique variables for every column at one time'''
    
    for col in df:
          print("The unique values for", col, "are:\n", df[col].unique(), '\n')

#********************************************************************************************************************

def cat_unique(df):
    categorical_col = []
    for column in df.columns:
        if df[column].dtype == object and len(df[column].unique()) <= 50:
            categorical_col.append(column)
            print(f"{column} : {df[column].unique()}")
            print("====================================")
            print()
    return categorical_col

#********************************************************************************************************************

def num_unique(df):
    numerical_col = []
    for column in df.columns:
        if df[column].dtype != object and len(df[column].unique()) <= 50:
            numerical_col.append(column)
            print(f"{column} : {df[column].unique()}")
            print("====================================")
            print()
    return numerical_col

#********************************************************************************************************************


def get_stats(data):
    statist = []  
    for col in data:
        min_var = data[col].min()
        mean_var = data[col].mean()
        max_var = data[col].max()
        list_metrics = [min_var, mean_var, max_var]
        statist.append(list_metrics)
        
    statist = pd.DataFrame(statist,columns = ['min', 'mean', 'max'], 
                           index = [data.columns])
    #print(statist)
    return statist

#********************************************************************************************************************

def scatter(df, name1, name2):
    
    plt.scatter(df[name2], df.name1, c = "seagreen", marker = "s")
    plt.title("Looking for correlations")
    plt.xlabel(name1)
    plt.ylabel(name2)
    plt.show()
    
    
