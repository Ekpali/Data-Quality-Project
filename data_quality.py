from sqlite3 import complete_statement
import streamlit as st
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import os

st.title('Data Quality Tool')

# Import dataset in either of the three accepted formats xlsx, csv or txt
# @st.experimental_memo(suppress_st_warning=True)
def file_uploader():    
    ''' This function uploads dataset into the system'''

    selected_file = st.sidebar.file_uploader("Please upload file", type=["xlsx", "csv", "txt"], accept_multiple_files=False)
    
    if selected_file is not None:
        
        if "csv" in selected_file.name:
            df = pd.read_csv(selected_file)
        elif "xlsx" in selected_file.name:
            df = pd.read_excel(selected_file)
        elif "txt" in selected_file.name:
            df = pd.read_fwf(selected_file)
        st.sidebar.success("upload successful")
    else:
        st.sidebar.info("Please upload a valid xlsx, csv or txt file")
        st.stop()
    
    return df 

# load dataset
dataset = file_uploader()

# Sidebar menus
# button to reload page 
# if st.sidebar.button("Load Data"):
#     file_uploader.clear()

sides = ["Data Quality Summary", "Single Column Analysis", "Multiple Column Analysis"]
options = st.sidebar.radio("Select Task", sides)

# DQ Summary
if options == "Data Quality Summary":
    st.subheader("Summary of Data Quality")

if options == 'Single Column Analysis':
    st.subheader("Single Column Analysis")

    # make dropdown list of each column 
    column_list = list(dataset)
    column_name = st.selectbox("Select Column to explore", column_list)

    Single_ops = ["Overview", "Missing Values", "Type Corrector", "Outliers"]
    ops = st.selectbox("Select Operation", Single_ops)

    if ops == "Missing Values":
        
        # Compute missing values
        missing_values = dataset[column_name].isnull().sum()
        total_values = len(dataset[column_name])

        missing_df = pd.DataFrame({"null":[missing_values], "total":[total_values]})
        # st.write(complete_df)

        # Preprocessing data exploration 
        pie, repair = st.columns(2)

        with pie:
            #make pie chart showing missing values percentage in selected column
            def pie_plotter(y, labels):
                fig = plt.figure()
                plt.pie(y, labels=labels)

                return st.pyplot(fig)

            pie_plotter(missing_df.iloc[0], missing_df.columns)

            # fig = plt.figure()
            # plt.pie(missing_df.iloc[0], labels=missing_df.columns)
            # st.pyplot(fig)
        
        with repair:
            miss_repair = st.radio("Select repair method", ["","Remove rows with null values", 
                                                            "Replace null values with mean", 
                                                            "Replace null values with nearest neighbour"])

            # remove missing 
            if miss_repair == "Remove rows with null values":
                missing_index = dataset[dataset[column_name].isnull()].index.tolist()
                null_actioned_df = dataset.drop(missing_index)
                null_actioned_df.reset_index()
            
            # replace missing with mean
            if miss_repair == "Replace null values with mean":
                if dataset[column_name].dtype == "int64":
                    missing_index = dataset[dataset[column_name].isnull()].index.tolist()
                    null_actioned_df = dataset[column_name].fillna(dataset[column_name].mean())
                else:
                    st.warning("Numeric column only")
