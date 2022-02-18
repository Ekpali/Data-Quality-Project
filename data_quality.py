import streamlit as st
import matplotlib as plt 
import pandas as pd

st.title('Data Quality Tool')

# Import dataset in either of the three accepted formats xlsx, csv or txt

def file_uploader():
    ''' This function uploads dataset into the system'''

    selected_file = st.file_uploader("", type=["xlsx", "csv", "txt"], accept_multiple_files=False)
    
    if selected_file is not None:
        
        if "csv" in selected_file.name:
            df = pd.read_csv(selected_file)
        elif "xlsx" in selected_file.name:
            df = pd.read_excel(selected_file)
        elif "txt" in selected_file.name:
            df = pd.read_fwf(selected_file)
        
        st.success("upload successful")
            
    else:
        st.info("Please upload a valid xlsx, csv or txt file")
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

    x1 = X_projected[:,0]
    x2 = X_projected[:,1]

    fig = plt.figure()
    plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar()

    st.pyplot(fig)
