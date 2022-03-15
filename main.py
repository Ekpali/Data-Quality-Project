import streamlit as st
from streamlit import caching
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import streamlit.components.v1 as components
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from collections import Counter
import os
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go

# outliers
from numpy import mean
from numpy import std
from scipy.stats import shapiro

st.set_page_config(layout = "wide")

st.title('Data Quality Tool')

# Clear cache to start fresh session
if st.sidebar.button("Start/Restart Session"):
    st.legacy_caching.caching.clear_cache()

# Import dataset in either of the three accepted formats xlsx, csv or txt
selected_file = st.sidebar.file_uploader("Please upload file", type=["xlsx", "csv", "txt"], accept_multiple_files=False) 

@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
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
main_sides = ["Data Summary", "Single Column Analysis", "Multiple Column Analysis"]
options = st.sidebar.radio("Select Task", main_sides)


########################################################################################
# DQ Summary #################
if options == "Data Summary":

    st.subheader('Data Summary Profile')

    ## start progress bar
    prog = st.progress(0)  
    r1, r2, r3, r4 = st.columns([2,1,1,1])
    with r4:
        download_report = st.empty()
    prof_succ = st.empty()

    ## Profiling function
    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def profile_reporter(dataset):
        return dataset.profile_report()

    prog.progress(10)

    profile = profile_reporter(dataset)

    prog.progress(30)

    st_profile_report(profile)

    ## end progress bar
    prog.progress(100)

    ## Download profile report
    with download_report.container():
        if st.button("Download Report"):
            profile.to_file("Data Summary Profile.html")
            cwd = os.getcwd()
            prof_succ.success('Report saved to ' + cwd)


#################################################################################################################################
#############################################    Single Column Analysis   #######################################################
#################################################################################################################################
if options == 'Single Column Analysis':
    st.subheader("Single Column Analysis")

    ## start progress bar
    prog = st.progress(0) 

    ## Dropdown list of type of analysis
    Single_ops = ["Missing Values", "Outliers", "Entry Type", "Duplicates", "Distributions"]
    ops = st.selectbox("Type of analysis", Single_ops)

    ## Container to hold dropdown list of each column
    col_holder = st.empty()

    ## Insert column list into container
    with col_holder.container():
        column_list = list(dataset)
        column_name = st.selectbox("Select column to analyse", column_list)

    ## end progress bar
    prog.progress(100)

    #define global delete button
    def delete_button(column_name):
        '''Delete column at any point'''
        ## remove column globally
        if st.button("Delete column"):
            dataset.drop(columns=[str(column_name)], axis=1, inplace=True)
            st.success('{} successfully removed'.format(str(column_name)))
            
            with col_holder.container():
                column_list = list(dataset)
                column_name = st.selectbox("Select column to analyse", column_list)

    ###############################################################################
    ## Missing values analysis ###################################################
    if ops == "Missing Values":
        st.subheader("Missing Values Identification and Repair")
        
        def compute_missing(dataset, column_name):
            '''compute missing values and save in dict'''
            missing_values = dataset[column_name].isnull().sum()
            total_values = len(dataset[column_name])
            comp_values = total_values - missing_values

            missing_df = {"Null":missing_values, "Not null":comp_values}

            return missing_df
        ### compute missing
        missing_df = compute_missing(dataset, column_name)

        def missing_values_plotter(val, k):
            '''plot missing values with pie chart'''
            #st.write(str(column_name))
            fig = go.Figure(data=[go.Pie(labels=k, values=val, textinfo='value+percent',
                             insidetextorientation='auto',rotation=90
                            )])

            colors = ['red', 'green']
            fig.update_traces(textfont_size=17,
                  marker=dict(colors=colors))
            fig.update_layout(margin=dict(t=30, b=0, l=0, r=80), title_text='Null values in {} '.format(str(column_name)), title_x=0.25)
            fig['layout']['title']['font'] = dict(size=20)
            
            return st.plotly_chart(fig, use_container_width=True)

        ### Preprocessing data exploration 
        pie, repair = st.columns([2,1])

        ### Column with placeholder for pie chart
        with pie:
            # make pie chart showing missing values percentage in selected column
            pie_holder = st.empty()
            
            with pie_holder.container():
                missing_values_plotter(list(missing_df.values()), list(missing_df.keys()))

        ## Check whether columns contain missing values
        if dataset[column_name].isnull().sum() != 0:

            ### Column with missing values repair options
            with repair:
                st.header('')
                st.markdown('Perform Repair Action')

                #### remove missing ###############################################################
                if st.button('Remove rows with null values'):        
                    missing_index = dataset[dataset[column_name].isnull()].index.tolist()
                    dataset.drop(missing_index, inplace=True)
                    dataset.reset_index()
                
                    with pie_holder.container():
                        missing_df = compute_missing(dataset, column_name)
                        missing_values_plotter(list(missing_df.values()), list(missing_df.keys()))
                    st.success('Null values successfully removed')

                #### replace missing values with mean  ############################################  
                if st.button("Replace null values with mean"):
                    if dataset[column_name].dtype == "int64" or dataset[column_name].dtype == "float64":
                        missing_index = dataset[dataset[column_name].isnull()].index.tolist()
                        dataset[column_name].fillna(dataset[column_name].mean(), inplace=True)

                        with pie_holder.container():
                            missing_df = compute_missing(dataset, column_name)
                            missing_values_plotter(list(missing_df.values()), list(missing_df.keys()))
                        st.success('Null values successfully replaced with mean values')

                    else:
                        st.warning("Numeric column only")

                #### replace null values with nearest neighbours #####################################
                if st.button("Replace null values with nearest neighbour"):
                    if dataset[column_name].dtype == "int64" or dataset[column_name].dtype == "float64":
                        int_data = dataset.select_dtypes(include=['int64', 'float64'])

                        #Normalise data
                        scaler = MinMaxScaler()
                        df_int = pd.DataFrame(scaler.fit_transform(int_data), columns = int_data.columns)

                        # Inpute values
                        imputer = KNNImputer(n_neighbors=15)
                        df_new = pd.DataFrame(imputer.fit_transform(df_int),columns = df_int.columns)

                        df_new[int_data.columns] = scaler.inverse_transform(df_new[int_data.columns])

                        dataset[int_data.columns] = df_new[int_data.columns]

                        with pie_holder.container():
                            missing_df = compute_missing(dataset, column_name)
                            missing_values_plotter(list(missing_df.values()), list(missing_df.keys()))
                        st.success('Null values successfully replaced with nearest neighbours')


                    else:
                        st.warning("Numeric column only")

                #### remove column ##############################################
                if st.button("Remove column"):
                    dataset.drop(columns=[str(column_name)], axis=1, inplace=True)
                    st.success('{} successfully removed'.format(str(column_name)))

                    with col_holder.container():
                        column_list = list(dataset)
                        column_name = st.selectbox("Select column to analyse", column_list)

                    with pie_holder.container():
                            missing_df = compute_missing(dataset, column_name)
                            missing_values_plotter(list(missing_df.values()), list(missing_df.keys()))
                                                
        else:
            with repair:
                st.info('There are no missing values in the column')
                #### remove column ##############################################
                if st.button("Remove column"):
                    dataset.drop(columns=[str(column_name)], axis=1, inplace=True)
                    st.success('{} successfully removed'.format(str(column_name)))

                    with col_holder.container():
                        column_list = list(dataset)
                        column_name = st.selectbox("Select column to analyse", column_list)

                    with pie_holder.container():
                            missing_df = compute_missing(dataset, column_name)
                            missing_values_plotter(list(missing_df.values()), list(missing_df.keys()))

    ##############################################################################
    ## Data type analysis #######################################################
    if ops == "Entry Type":
        st.subheader("Entry Type Explorer")

        def float_digit(n: str) -> bool:
            try:
                float(n)
                return True
            except ValueError:
                return False

        ### function to determine data types and indexes 
        def compute_datatype(dataset):
            '''This function determines datatypes and indexes'''
            global list_dtypes, digit_index, str_index, other_index

            ### Lists that hold data types and indexes
            list_dtypes = []
            digit_index = []
            str_index = []
            other_index = []

            for index, element in enumerate(dataset[column_name]):
                if (str(element).isnumeric()) or float_digit(str(element)) == True:
                    list_dtypes.append('numeric')
                    digit_index.append(index)
                elif (str(element).isalpha()):
                    list_dtypes.append('string')
                    str_index.append(index)
                else:
                    list_dtypes.append('other')
                    other_index.append(index)

            uniq_list_dtypes = Counter(list_dtypes).keys()
            uniq_counts = Counter(list_dtypes).values()

            return uniq_list_dtypes, uniq_counts

        ### funtion to make barplot
        def barplotter(list, count):
            fig = px.bar(x=list, y=count, labels= {'y': 'entry count', 'x': 'entry type'}, 
                        title='Entry types in {} '.format(str(column_name)))
            fig.update_layout(margin=dict(t=30, b=0, l=0, r=80), title_x=0.5)
            #fig.update_traces(textfont_size=30)
            fig['layout']['title']['font'] = dict(size=20)
            return st.plotly_chart(fig, use_container_width=True)
 

        ### Columns to hold barplot and remedy
        bar, rem = st.columns([2,1])

        ### column to hold datatype info and barplot
        with bar:
            bar_holder = st.empty()
            uniq_list_dtypes, uniq_counts = compute_datatype(dataset)
           
            with bar_holder.container():
                barplotter(uniq_list_dtypes, uniq_counts)
        
        ### column holding remedy buttons 
        with rem:
            st.subheader('')
            st.write('Perform repair action')
            if st.button('Delete numeric entries'):
                dataset.drop(digit_index, inplace=True)
                dataset.reset_index()
                uniq_list_dtypes, uniq_counts = compute_datatype(dataset)
                st.success('numeric enteries removed')
                with bar_holder.container():
                    barplotter(uniq_list_dtypes, uniq_counts)

            if st.button('Delete string entries'):
                dataset.drop(str_index, inplace=True)
                dataset.reset_index()
                uniq_list_dtypes, uniq_counts = compute_datatype(dataset)
                st.success('string entries removed')
                with bar_holder.container():
                    barplotter(uniq_list_dtypes, uniq_counts)

            if st.button('Delete other entries'):
                dataset.drop(other_index, inplace=True)
                dataset.reset_index()
                uniq_list_dtypes, uniq_counts = compute_datatype(dataset)
                st.success('string entries removed')
                with bar_holder.container():
                    barplotter(uniq_list_dtypes, uniq_counts)                 

    ################################################################################################
    ## Outlier Ananlysis############################################################################
    if ops == "Outliers": # codes from https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/ and 
                           # https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
        
        st.subheader("Outlier identification and removal")

        with col_holder.container():
            column_list = dataset.select_dtypes(include=['int64', 'float64']).columns
            column_name = st.selectbox("Select column to analyse", column_list)
            st.info('Showing numeric columns only')

        box, outlier_repair = st.columns([2,1])

        def box_plotter(dataset):
            '''This function makes a box plot'''
            fig = px.box(dataset[column_name])
            fig.update_layout(margin=dict(t=30, b=0, l=0, r=80), title_text='Boxplot of {} '.format(str(column_name)), title_x=0.5)

            fig['layout']['title']['font'] = dict(size=20)
            
            return st.plotly_chart(fig, use_container_width=True)
        
        ### column holding boxplot
        with box:
            box_plt = st.empty()

            with box_plt.container():
                box_plotter(dataset)
            
            #### suggestion area ------------------------
            st.write('Key statistics')
            alpha = 0.05
            stat, p = shapiro(dataset[column_name])
            out_stat = pd.DataFrame([{'CI':'5%','p-value':p, 'statistic':stat}])
            st.write(out_stat)

        ### column holding repair methods
        with outlier_repair:
            st.subheader('')
            st.write('Select repair method')

            ##### show dataframe of key values
            def show_outlier_df(lower, upper):
                global outliers 
                outliers = [x for x in dataset[column_name] if x < lower or x > upper]
                out_df = pd.DataFrame({'No of outliers': len(outliers), 'Lower': lower, 'Upper': upper}, index=[0])
                st.write(out_df)
            
            ##### update boxplot
            def update_boxplot():
                with box_plt.container():
                    box_plotter(dataset)

            ##### print success message
            def outlier_success_print():
                if len(outliers) > 0:
                    st.success('{} removed successfully'.format(len(outliers)))

                else:
                    st.info('No identified outliers')

            #### show outliers in df, remove outliers and update boxplot
            def outlier_removal_update(lower, upper):
                    #show outliers
                    show_outlier_df(lower, upper)
                    ##### identify and remove outliers
                    out_idx = dataset[(dataset[column_name] < lower) | (dataset[column_name] > upper)].index.tolist()
                    dataset.drop(out_idx, inplace=True)
                    dataset.reset_index()
                    ##### update boxplot
                    update_boxplot()
                    ##### print
                    outlier_success_print()

            #### repair using standard deviation
            if st.button('Standard deviation'):

                ##### calculate summary statistics
                data_mean, data_std = mean(dataset[column_name]), std(dataset[column_name])
                ##### identify outliers
                cut_off = data_std * 3
                lower, upper = data_mean - cut_off, data_mean + cut_off
                ##### show results
                outlier_removal_update(lower, upper)
                

            #### repair using interquartile range
            if st.button('Interquartile range'):

                ##### calculate interquartile range
                q25, q75 = np.percentile(dataset[column_name], 25), np.percentile(dataset[column_name], 75)
                iqr = q75 - q25
                ##### calculate the outlier cutoff
                cut_off = iqr * 1.5
                lower, upper = q25 - cut_off, q75 + cut_off
                ##### show results
                outlier_removal_update(lower, upper)

            #### repair outliers using user inpute
            with st.expander('User input'):
                lower = st.number_input("lower bound", min_value=0, step=10)
                upper = st.number_input("upper bound", min_value=0, step=10)

                if st.button('Execute'):  
                    ##### show results
                    outlier_removal_update(lower, upper)

            

            


# delete_holder = st.empty()
# with delete_holder.container():
#     delete_button(column_name)

#####################################################################################################################################
# Multiple Column Analysis #####
if options == 'Multiple Column Analysis':
    st.subheader("Multiple Column Analysis")

    ## start progress bar
    prog = st.progress(0)

    multi_ops = ["Missing values", "Clusters and Outliers", "Transpose", "Anomaly Detection"]
    ops = st.selectbox("Select Analysis", multi_ops)

    if multi_ops == "Missing Values":
        dataset.isna().any()

        dataset.isna().sum()

    ## end progress bar
    prog.progress(100)


####################################################################################################################################
# Download dataset after processing
data = dataset.to_csv(index=False).encode('utf-8')
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

# if __name__ == '__main__':
#     ...
