from textwrap import fill
import streamlit as st
from streamlit import caching
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import streamlit.components.v1 as components
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from collections import Counter
from ipywidgets import widgets

st.set_page_config(layout = "wide")

st.title('Data Quality Tool')

# Clear cache to start fresh session
if st.sidebar.button("Start/Restart Session"):
    st.legacy_caching.caching.clear_cache()

# Import dataset in either of the three accepted formats xlsx, csv or txt
selected_file = st.sidebar.file_uploader("Please upload file", type=["xlsx", "csv", "txt"], accept_multiple_files=False) 

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def file_uploader():    
    ''' This function uploads dataset into the system'''
    
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


#############################################################################################################################################
## Main functions (contains a sidebar of the compatible functions) ###########
main_sides = ["Data Summary", "Single Column Analysis", "Multiple Column Analysis", "General Analysis"]
options = st.sidebar.radio("Select Task", main_sides)


########################################################################################
## DQ Summary #################
if options == "Data Summary":
    # download_report = st.empty()

    profile = dataset.profile_report()
    # profile = ProfileReport(dataset, explorative=True, minimal=False)
    st_profile_report(profile)
    #components.html(profile, height=1000, width=1050,scrolling=True)

    # with download_report.container():
    #     if st.button("Download Report"):
    #         #profile.to_file("Data_profile.html")

#################################################################################################################################
# Single Column Analysis ######
if options == 'Single Column Analysis':
    st.subheader("Single Column Analysis")

    ## Dropdown list of type of analysis
    Single_ops = ["Missing Values", "Data Type", "Outliers", "Duplicates"]
    ops = st.selectbox("Type of analysis", Single_ops)

    ## Container to hold dropdown list of each column
    col_holder = st.empty()

    ## Insert column list into container
    with col_holder.container():
        column_list = list(dataset)
        column_name = st.selectbox("Select column to analyse", column_list)


    #############################################################
    ## Missing values analysis ##################################
    if ops == "Missing Values":
        st.subheader("Missing Values Identification and Repair")
        
        def compute_missing(dataset, column_name):
            '''compute missing values and save in dict'''
            missing_values = dataset[column_name].isnull().sum()
            total_values = len(dataset[column_name])
            comp_values = total_values - missing_values

            missing_df = {"Null":missing_values, "Not null":comp_values}

            return missing_df

        missing_df = compute_missing(dataset, column_name)

        def missing_values_plotter(y, labels):
            '''plot missing values with pie chart'''
            fig = plt.figure()
            plt.title(str(column_name))
            plt.pie(y, labels=labels, autopct=lambda p:f'{p:.2f}% ({p*sum(list(missing_df.values()))/100 :.0f})', colors=['red', 'green'] )
            plt.axis('equal')
            plt.legend(loc='lower right')

            return st.pyplot(fig)


        ### Preprocessing data exploration 
        pie, repair = st.columns(2)

        ### Column with placeholder for pie chart
        with pie:
            # make pie chart showing missing values percentage in selected column
            pie_holder = st.empty()
            
            with pie_holder.container():
                missing_values_plotter(list(missing_df.values()), labels=list(missing_df.keys()))

        ## Check whether columns contain missing values
        if dataset[column_name].isnull().sum() != 0:

            ### Column with missing values repair options
            with repair:
                st.markdown('Perform Repair Action')
                #### remove missing 
                if st.button('Remove rows with null values'):        
                    missing_index = dataset[dataset[column_name].isnull()].index.tolist()
                    dataset.drop(missing_index, inplace=True)
                    dataset.reset_index()
                
                    with pie_holder.container():
                        missing_df = compute_missing(dataset, column_name)
                        missing_values_plotter(list(missing_df.values()), labels=list(missing_df.keys()))
                    st.success('Null values successfully removed')

                #### replace missing values with mean    
                if st.button("Replace null values with mean"):
                    if dataset[column_name].dtype == "int64" or dataset[column_name].dtype == "float64":
                        missing_index = dataset[dataset[column_name].isnull()].index.tolist()
                        dataset[column_name].fillna(dataset[column_name].mean(), inplace=True)

                        with pie_holder.container():
                            missing_df = compute_missing(dataset, column_name)
                            missing_values_plotter(list(missing_df.values()), labels=list(missing_df.keys()))
                        st.success('Null values successfully replaced')

                    else:
                        st.warning("Numeric column only")

                #### replace null values with nearest neighbours
                if st.button("Replace null values with nearest neighbour"):
                    st.container()

                #### remove column
                if st.button("Remove column"):
                    dataset.drop(columns=[str(column_name)], axis=1, inplace=True)
                    st.success('{} successfully removed'.format(str(column_name)))
                    

                    with col_holder.container():
                        column_list = list(dataset)
                        column_name = st.selectbox("Select column to analyse", column_list)

                    with pie_holder.container():
                            missing_df = compute_missing(dataset, column_name)
                            missing_values_plotter(list(missing_df.values()), labels=list(missing_df.keys()))
                            
                    
                    
                        
        else:
            with repair:
                st.info('There are no missing values in the column')

    ##############################################################################
    ## Data type analysis
    if ops == "Data Type":
        st.subheader("Entry Type Explorer")

        list_dtypes = []
        for element in dataset[column_name]:
            if isinstance(element, int):
                list_dtypes.append('int')
            elif isinstance(element, float):
                list_dtypes.append('float')
            elif isinstance(element, str):
                list_dtypes.append('str')

        uniq_list_dtypes = Counter(list_dtypes).keys()
        uniq_counts = Counter(list_dtypes).values()
        no_dtypes = len(uniq_list_dtypes)

        ### Columns to hold barplot and remedy
        bar, rem = st.columns(2)


        with bar:
            ##### check whether column contains multiple datatypes
            if no_dtypes == 1:
                st.info("All enteries in this column are of DataType: {}".format(str(set(list_dtypes))[1:-1]))


            else:
                st.info("The column has multiple datatypes: {}".format(str(set(list_dtypes))[1:-1]))
                
                def barplotter(x, y):
                    fig = plt.figure()
                    plt.bar(x, y)
                    plt.xlabel('datatypes')
                    plt.ylabel('frequency')
                    
                    return st.pyplot(fig)

                barplotter(uniq_list_dtypes, uniq_counts)
        
        #with rem:




                


#####################################################################################################################################
# Multiple Column Analysis #####
if options == 'Multiple Column Analysis':
    st.subheader("Multiple Column Analysis")

    multi_ops = ["Missing values", "Correlations", "Clusters and Outliers", "Summaries and Sketches"]
    ops = st.selectbox("Select Analysis", multi_ops)

##################################################################################
# General Analysis #####
if options == 'General Analysis':
    st.subheader("General Analysis")


####################################################################################################################################
# Download dataset after processing
data = dataset.to_csv().encode('utf-8')
st.sidebar.header('')
st.sidebar.header('')
st.sidebar.header('')
st.sidebar.header('')
st.sidebar.header('')
st.sidebar.header('')
st.sidebar.header('')
st.sidebar.header('')

st.sidebar.download_button(
    label="Download data as CSV ⬇️",
    data=data,
    file_name='cleaned_data.csv',
    mime='text/csv',
)

