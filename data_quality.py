import streamlit as st
import matplotlib as plt 
import pandas as pd

st.title('Data Quality Tool')

# Import dataset in either of the three accepted formats xlsx, csv or txt

def file_uploader():
    ''' This function uploads dataset into the system'''

    selected_file = st.sidebar.file_uploader("", type=["xlsx", "csv", "txt"], accept_multiple_files=False)
    
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

dataset = file_uploader()

# Preprocessing data exploration 
e1, e2 = st.columns(2)

data_head = e1.button("view head")
if  data_head:
    st.write(dataset.head())
    
data_summary = e2.button("View Summary Statistics")
if  data_summary:
    st.write(dataset.describe())


# Sidebar menus
sides = ["Load Data", "Quality Check", "EDA", "ML"]

options = st.sidebar.radio("Select Task", sides)

if options == 'Quality Check':
    st.subheader("Welcome to Data Quality Exploration")
