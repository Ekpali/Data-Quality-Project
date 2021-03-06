import streamlit as st
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import openpyxl
from collections import Counter
import os
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import base64
from PIL import Image



# outliers
from numpy import mean
from numpy import std
from scipy.stats import shapiro
from zmq import select

# configure page with
st.set_page_config(layout = "wide")


# css styling where needed
# with open('style.css') as f:
#     st.markdown(f""" <style>{f.read()}</style>""", unsafe_allow_html=True)

# # Set background image courtesy of soft-nougat on https://discuss.streamlit.io/t/how-do-i-use-a-background-image-on-streamlit/5067/17
# def set_bg_hack(main_bg):
#     '''
#     A function to unpack an image from root folder and set as bg.
 
#     Returns
#     -------
#     The background.
#     '''
#     # set bg name
#     main_bg_ext = "png"
        
#     st.markdown(
#          f"""
#          <style>
#          .stApp {{
#              background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
#              background-size: cover
#          }}
#          </style>
#          """,
#          unsafe_allow_html=True
#      )

# set_bg_hack('background.jpg')

# # Set side image courtesy of soft-nougat on https://discuss.streamlit.io/t/how-do-i-use-a-background-image-on-streamlit/5067/17
# def sidebar_bg(side_bg):

#    side_bg_ext = 'png'

#    st.markdown(
#       f"""
#       <style>
#       [data-testid="stSidebar"] > div:first-child {{
#           background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
#       }}

#       </style>
#       """,
#       unsafe_allow_html=True,
#       )

# side_bg = 'sidecolor.png'
# sidebar_bg(side_bg)

# Main heading
st.title('Data Profiling and Quality Analysis Tool')

intro_Section = st.empty()



# into page function
def intro():
    st.subheader('Welcome to the data exploration, quality analysis and repair tool ????')
    description, flow = st.columns([2.5,1])
    with description:
        st.markdown(f"""[![Ekpali - Data-Quality-Project](https://img.shields.io/static/v1?label=Ekpali&message=Data-Quality-Project&color=blue&logo=github)](https://github.com/Ekpali/Data-Quality-Project "Go to GitHub repo")""")
        st.write('This tool supports the following features:')
        """ 
        * Data profiling using Pandas Profiler (highly recommended step prior to further analysis)
        * Single Column Analysis
            * Missing value identification, removal and imputation 
            * Identification and removal of Outliers
            * Column entry type exploration and
            * Column distributions for both categorical and numerical features
            * Other features: range handling and splitting columns
        * Multiple columns Analysis
            * Missing value identification, removal and imputation
            * Scatter plots and distributions
            * Duplicates detection and removal
            * Correlation exploration 
            * Anomaly detection 
            * Clusters
        """
    with flow:
        image = Image.open('flow.png')
        st.image(image, width=None)
    st.info('Clicking the Restart Session button clears the cache of the system, undoing all perfomed steps')

# header section for single/multiple column analysis
header_section = st.empty()

#########################################################################################################################################
################################### Start of main functionalities #######################################################################

# Clear cache to start fresh session
if st.sidebar.button("Restart Session"):
    st.legacy_caching.caching.clear_cache()
    for hist_holder in st.session_state.keys():
        del st.session_state[hist_holder]
    for data_store in st.session_state.keys():
        del st.session_state[data_store]

# main columns for images and repair functionalities 
img, repair_options, hist, undo = st.columns([2.5,1.1,1.1,0.4])


# Import dataset in either of the three accepted formats xlsx, csv or txt
selected_file = st.sidebar.file_uploader("Please upload file", type=["xlsx", "csv", "txt"], accept_multiple_files=False)

# load dataset
data_holder = st.empty()

# Function to import dataset
@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def format_uploader():
    ''' This function converts dataset to dataframe'''
    if selected_file is not None:
        with intro_Section.container():
            st.empty()
        
        if "csv" in selected_file.name:
            df = pd.read_csv(selected_file)
        elif "xlsx" in selected_file.name:
            df = pd.read_excel(selected_file)
        elif "txt" in selected_file.name:
            df = pd.read_csv(selected_file, sep='\t')
        st.sidebar.success("upload successful")

    else:
        st.sidebar.info("Please upload a valid xlsx, csv or txt file")
        with data_holder.container():
            st.empty()
        with intro_Section.container():
            intro()
        # st.legacy_caching.caching.clear_cache()
        # for hist_holder in st.session_state.keys():
        #     del st.session_state[hist_holder]
        # for data_store in st.session_state.keys():
        #     del st.session_state[data_store]
        st.stop()

    return df 


# intialize data storage for undoing changes
if 'data_store' not in st.session_state:
    st.session_state.data_store = []



with data_holder.container():   
    dataset = format_uploader()

# append copy of dataset to session state before carrying out Options
def append_data(data_copy):
    st.session_state.data_store.append(data_copy)


#############################################################################################################################################
## Main functions (contains a sidebar of the compatible functions) ###########
st.sidebar.header('')
with st.sidebar.expander('Select Task', expanded=True): 
    main_options = st.radio("", ["Data Profile/Summary", "Single Column Analysis", "Multiple Column Analysis"])

# see recommended steps
with st.sidebar.expander('Show recommended steps'):
    image1 = Image.open('flow.png')
    st.image(image1, width=None)

########################################################################################
# DQ Summary #################
if main_options == "Data Profile/Summary":

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
    ## create profile
    profile = profile_reporter(dataset)
    prog.progress(30)
    ## show profile
    st.empty()
    st.empty()
    st_profile_report(profile)

    ## end progress bar
    prog.progress(100)

    ## Download profile report
    with download_report.container():
        if st.button("Download Report"):
            profile.to_file("Data Summary Profile.html")
            cwd = os.getcwd()
            prof_succ.success('Report saved to ' + cwd)


# load history into session state
if 'hist_holder' not in st.session_state:
    st.session_state.hist_holder = []

# delete history buttion 
def history_button():
    '''delete history from session state'''
    global dataset
    with hist:
        st.markdown('History')
        if st.button('Clear history'):
            if len(st.session_state.hist_holder) > 0:
                st.legacy_caching.caching.clear_cache()
                for hist_holder in st.session_state.keys():
                    del st.session_state[hist_holder]
                for data_store in st.session_state.keys():
                    del st.session_state[data_store]
                with data_holder.container():   
                    dataset = format_uploader()
                
            else:
                st.warning('No recorded history')
        
    if 'hist_holder' not in st.session_state:
        st.session_state.hist_holder = []
        

def undo_button():
    with undo:
        st.write('Undo')
        if st.button('??????'):
            #st.legacy_caching.caching.clear_cache()
            if len(st.session_state.hist_holder) > 0:
                st.session_state.hist_holder.pop()
                with data_holder.container():
                    store_df = pd.DataFrame(st.session_state.data_store[-1]).copy(deep=True)
                    if len(dataset.columns.to_list()) != len(store_df.columns.to_list()):
                        non_df = np.setdiff1d(store_df.columns, dataset.columns)
                        for index, value in enumerate(non_df):
                            dataset.insert(store_df.columns.get_loc(non_df[index]), non_df[index], store_df.pop(non_df[index]))
                    else:
                        dataset.drop(dataset.index, inplace=True)
                        dataset[dataset.columns.to_list()] = store_df[store_df.columns.to_list()]
                
                st.session_state.data_store.pop()


            else:
                st.warning('Null')


# print all history
def input_hist():
    '''print history in sidebar'''
    with hist:
        
        for item in st.session_state.hist_holder:
            history = f'<p style="color: rgba(61, 140, 224); font-size: 13px; border-radius: 4px; padding: 5px; background-color: rgba(12, 48, 94, 0.2);">{item}</p>'
            st.markdown(history, unsafe_allow_html=True)
    


#################################################################################################################################
#############################################    Single Column Analysis   #######################################################
#################################################################################################################################
if main_options == 'Single Column Analysis':
    history_button()
    undo_button()

    with header_section.container():
        st.subheader("Single Column Analysis")

        ### start progress bar
        prog = st.progress(0) 

        ### Dropdown list of type of analysis
        ops = st.selectbox("Select Analysis", ["Missing Values", "Outliers", "Entry Type", "Distributions", "Other"])
        #st.write(st.session_state.data_store)

        ### end progress bar
        prog.progress(100)

        ### Container to hold dropdown list of each column
        col_holder = st.empty()

        ### Insert column list into container
        with col_holder.container():
            column_list = list(dataset)
            column_name = st.selectbox("Select Column", column_list)

        ### show column type
        coltype_holder = st.empty()
        with coltype_holder.container():
            data_conv = dataset.convert_dtypes()
            col_type = data_conv[column_name].dtype
            st.info('{} column type is: {}'.format(column_name, col_type))

        ### update coltype after column removal
        def update_coltype():
            with coltype_holder.container():
                data_conv = dataset.convert_dtypes()
                col_type = data_conv[column_name].dtype
                st.info('{} column type is: {}'.format(column_name, col_type))

        

    ###############################################################################
    ## Missing values analysis ###################################################
    if ops == "Missing Values":
        #st.subheader("Missing Values Identification and Repair")
        
        ### compute missing values and place them in dict
        def compute_missing(dataset, column_name):
            '''compute missing values and save in dict'''
            missing_values = dataset[column_name].isnull().sum()
            total_values = len(dataset[column_name])
            comp_values = total_values - missing_values

            missing_df = {"Null":missing_values, "Not null":comp_values}

            return missing_df
        ### compute missing
        missing_df = compute_missing(dataset, column_name)
        
        ### plot missing values using piechart
        def missing_values_plotter(val, k):
            '''plot missing values with pie chart'''
            #st.write(str(column_name))
            fig = go.Figure(data=[go.Pie(labels=k, values=val, textinfo='value+percent',
                             insidetextorientation='auto',rotation=90
                            )])

            colors = ['lavender', 'blue']
            fig.update_traces(textfont_size=17,
                  marker=dict(colors=colors))
            fig.update_layout(margin=dict(t=30, b=0, l=0, r=10), title_text='Null values in {} '.format(str(column_name)), title_x=0.25)
            fig['layout']['title']['font'] = dict(size=20)
            
            return st.plotly_chart(fig, use_container_width=True) 

        def recompute_and_plot():
            '''recompute missing values and update plot'''
            with pie_holder.container():
                missing_df = compute_missing(dataset, column_name)
                missing_values_plotter(list(missing_df.values()), list(missing_df.keys()))

        ### Column with placeholder for pie chart
        with img:
            # make pie chart showing missing values percentage in selected column
            pie_holder = st.empty()
            
            with pie_holder.container():
                missing_values_plotter(list(missing_df.values()), list(missing_df.keys()))

        ## Check whether columns contain missing values
        if dataset[column_name].isnull().sum() != 0:

            ### Column with missing values repair options
            with repair_options:
                #st.header('')
                st.markdown('Options')

                #### remove missing ###############################################################
                if st.button('Remove null values'):        
                    missing_index = dataset[dataset[column_name].isnull()].index.tolist()

                    d_copy = dataset.copy(deep=True)
                    append_data(d_copy)
                    
                    dataset.drop(missing_index, inplace=True)
                    dataset.reset_index(drop=True, inplace=True)
                
                    recompute_and_plot()
                    st.success('Null values successfully removed')
                    ####### add to history tab
                    st.session_state.hist_holder.append(f'Null removed from {column_name}')
                    
                #### replace missing values with mean  ############################################  
                if st.button("Replace null with mean"):
                    if dataset[column_name].dtype == "int64" or dataset[column_name].dtype == "float64":
                        
                        d_copy = dataset.copy(deep=True)
                        append_data(d_copy)

                        dataset[column_name].fillna(dataset[column_name].mean(), inplace=True)

                        recompute_and_plot()
                        st.success('Null values successfully replaced with mean values')
                        ####### add to history tab
                        st.session_state.hist_holder.append(f'Null replaced with mean in {column_name}')

                    else:
                        st.warning("Numeric column only")

                #### replace null values with nearest neighbours #####################################
                with st.expander('Replace null with KNN'):
                    if dataset[column_name].dtype == "int64" or dataset[column_name].dtype == "float64":
                        k_n = st.number_input('n_neighbours', min_value=0)
                        if k_n > 0:
                            if st.button("Replace null"):
                                
                                    int_data = dataset.select_dtypes(include=['int64', 'float64'])

                                    d_copy = dataset.copy(deep=True)
                                    append_data(d_copy)

                                    #Normalise data
                                    scaler = MinMaxScaler()
                                    df_int = pd.DataFrame(scaler.fit_transform(int_data), columns = int_data.columns)

                                    # Inpute values
                                    imputer = KNNImputer(n_neighbors=5)
                                    df_new = pd.DataFrame(imputer.fit_transform(df_int),columns = df_int.columns)

                                    df_new[int_data.columns] = scaler.inverse_transform(df_new[int_data.columns])

                                    dataset[column_name] = df_new[column_name]

                                    recompute_and_plot()
                                    st.success('Null values successfully replaced with nearest neighbours')
                                    ####### add to history tab
                                    st.session_state.hist_holder.append(f'Null replaced with nearest neighbours (k = {k_n}) in {column_name}')


                    else:
                        st.warning("Numeric column only")

                #### repair missing value using user inpute ################################################
                with st.expander('Replace null with input'):
                    if dataset[column_name].dtype == "int64" or dataset[column_name].dtype == "float64":
                        user_miss_input = st.number_input("Input number", min_value=0)
                
                    else:
                        user_miss_input = st.text_input("Input text")

                    if st.button('Replace'):

                        d_copy = dataset.copy(deep=True)
                        append_data(d_copy)

                        dataset[column_name].fillna(value=user_miss_input, inplace=True)
                        ##### show results
                        recompute_and_plot()
                        st.success('Null values successfully replaced with user input')
                        ####### add to history tab
                        st.session_state.hist_holder.append(f'Null replaced with user input in {column_name}')

                #### remove column ##############################################
                if st.button("Remove column"):
                    ####### add to history tab
                    st.session_state.hist_holder.append(f'{column_name} removed')

                    d_copy = dataset.copy(deep=True)
                    append_data(d_copy)

                    dataset.drop(columns=[str(column_name)], axis=1, inplace=True)
                    st.success('{} successfully removed'.format(str(column_name)))
                    #check_data(d_test)

                    with col_holder.container():
                        column_list = list(dataset)
                        column_name = st.selectbox("Select column to analyse", column_list)
                        st.info('{} column type is: {}'.format(column_name, col_type))

                    with coltype_holder.container():
                        st.empty()
                    recompute_and_plot()
                    
                                                
        else:
            # with img:
            #     st.info('No missing values in the column')
            with repair_options:
                st.write('Options')
                #### remove column ##############################################
                if st.button("Remove column"):

                    d_copy = dataset.copy(deep=True)
                    append_data(d_copy)

                    dataset.drop(columns=[str(column_name)], axis=1, inplace=True)
                    st.success('{} successfully removed'.format(str(column_name)))
                    ####### add to history tab
                    st.session_state.hist_holder.append(f'{column_name} removed')

                    with col_holder.container():
                        column_list = list(dataset)
                        column_name = st.selectbox("Select column to analyse", column_list)
                        st.info('{} column type is: {}'.format(column_name, col_type))

                    with coltype_holder.container():
                        st.empty()
                    recompute_and_plot()
                    
    

    ##############################################################################
    ## Data type analysis #######################################################
    if ops == "Entry Type":
        
        #st.subheader("Entry Type Explorer")

        def float_digit(n: str) -> bool:
            try:
                float(n)
                return True
            except ValueError:
                return False

        ### function to determine data types and indexes 
        def compute_datatype():
            dataset.reset_index(drop=True, inplace=True)
            #'''This function determines datatypes and indexes'''
            global uniq_list_dtypes, uniq_counts, list_dtypes, digit_index, str_index, other_index

            ### Lists that hold data types and indexes
            list_dtypes = []
            digit_index = []
            str_index = []

            for index, element in enumerate(dataset[column_name]):
                if (str(element).isnumeric()) or float_digit(str(element)) == True:
                    list_dtypes.append('numeric')
                    digit_index.append(index)
                else:
                    list_dtypes.append('string')
                    str_index.append(index)

            uniq_list_dtypes = Counter(list_dtypes).keys()
            uniq_counts = Counter(list_dtypes).values()

        ### funtion to make barplot
        def barplotter(list, count):
            fig = px.bar(x=list, y=count, labels= {'y': 'entry count', 'x': 'entry type'}, 
                        title='Entry types in {} '.format(str(column_name)))
            fig.update_layout(margin=dict(t=30, b=0, l=0, r=80), title_x=0.5)
            #fig.update_traces(textfont_size=30)
            fig['layout']['title']['font'] = dict(size=20)
            return st.plotly_chart(fig, use_container_width=True)

        ### column to hold datatype info and barplot
        
        with img:
            if dataset[column_name].isna().sum().sum() > 0:
                    st.warning('Column contains missing values that may cause misinterpretation, please handle them first')
            bar_holder = st.empty()
            #compute_datatype()
           
            with bar_holder.container():
                compute_datatype()
                barplotter(uniq_list_dtypes, uniq_counts)
        
        ### column holding remedy buttons 
        with repair_options:
            compute_datatype()
            st.write('Options')

            if len(uniq_list_dtypes) > 1:

                #### delete numeric entries ###############################################################
                if st.button('Delete numeric entries'):
                    if len(digit_index) > 0:
                        d_copy = dataset.copy(deep=True)
                        append_data(d_copy)

                        dataset.drop(digit_index, inplace=True)
                        compute_datatype()
                        st.success('numeric enteries removed')
                        with bar_holder.container():
                            barplotter(uniq_list_dtypes, uniq_counts)
                        ####### add to history tab
                        st.session_state.hist_holder.append(f'{column_name} numerics removed')
                    else:
                        st.info('No numeric enteries found')

                #### delete string entries ################################################################
                if st.button('Delete string entries'):
                    if len(str_index) > 0:
                        d_copy = dataset.copy(deep=True)
                        append_data(d_copy)
                        dataset.drop(str_index, inplace=True)
                        compute_datatype()
                        st.success('string entries removed')
                        with bar_holder.container():
                            barplotter(uniq_list_dtypes, uniq_counts)
                        ####### add to history tab
                        st.session_state.hist_holder.append(f'{column_name} strings removed')
                    else:
                        st.info('No string enteries found')

                

            elif len(uniq_list_dtypes) == 1:
                st.info('Column has one entry type')

            else: 
                st.error('Unable to identify type of entries in this column')
            
            # with st.expander('Convert column type'):
            #         if dataset[column_name].isna().sum().sum() > 0:
            #             st.warning('Column contains NaN values, please handle')
            #         else:
            #             if st.button('Convert to numeric'):
            #                 d_copy = dataset.copy(deep=True)
            #                 append_data(d_copy)

            #                 dataset[column_name] = dataset[column_name].convert_dtypes()

            #                 st.session_state.hist_holder.append(f'{column_name} converted to numeric')


    ################################################################################################
    ## Outlier Ananlysis############################################################################
    if ops == "Outliers": # codes from https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/ and 
                           # https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
        
        #st.subheader("Outlier identification and removal")

        with col_holder.container():
            column_list = dataset.select_dtypes(include=[np.number]).columns
            column_name = st.selectbox("Select column (numeric only)", column_list)
    
        if len(list(column_list)) > 0:

            ### update coltype_holder
            update_coltype()

            def box_plotter(dataset):
                '''This function makes a box plot'''
                fig = px.box(dataset[column_name])
                fig.update_layout(margin=dict(t=30, b=0, l=0, r=10), title_text='Boxplot of {} '.format(str(column_name)), title_x=0.5)

                fig['layout']['title']['font'] = dict(size=20)
                
                return st.plotly_chart(fig, use_container_width=True)
        
            ### column holding boxplot
            with img:
                box_plt = st.empty()

                with box_plt.container():
                    box_plotter(dataset)
                
                #### suggestion area 
                st.write('Key statistics')
                alpha = 0.05
                stat, p = shapiro(dataset[column_name])
                out_stat = pd.DataFrame([{'CI':'5%','p-value':p, 'statistic':stat}])
                st.write(out_stat)
                if dataset[column_name].isnull().sum() > 0:
                    st.warning('Column contains null values causing NA in statistic measure')

            ### column holding repair methods
            with repair_options:
                st.write('Options')

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
                        st.success('{} outliers removed successfully'.format(len(outliers)))

                    else:
                        st.info('No identified outliers')

                #### show outliers in df, remove outliers and update boxplot
                def outlier_removal_update(lower, upper):
                        #show outliers
                        show_outlier_df(lower, upper)
                        ##### identify and remove outliers
                        out_idx = dataset[(dataset[column_name] < lower) | (dataset[column_name] > upper)].index.tolist()

                        d_copy = dataset.copy(deep=True)
                        append_data(d_copy)

                        dataset.drop(out_idx, inplace=True)
                        dataset.reset_index(drop=True, inplace=True)
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
                    if len(outliers) > 0:
                        ####### add to history tab
                        st.session_state.hist_holder.append(f'{len(outliers)} outliers removed from {column_name} using StD')
                    

                #### repair using interquartile range
                if st.button('Interquartile range'):

                    ##### calculate interquartile range
                    q25, q75 = np.quantile(dataset[column_name], 0.25), np.quantile(dataset[column_name], 0.75)
                    iqr = q75 - q25
                    ##### calculate the outlier cutoff
                    cut_off = iqr * 1.5
                    lower, upper = q25 - cut_off, q75 + cut_off
                    ##### show results
                    outlier_removal_update(lower, upper)
                    if len(outliers) > 0:
                        ####### add to history tab
                        st.session_state.hist_holder.append(f'{len(outliers)} outliers removed from {column_name} using IQR')

                #### repair outliers using user inpute
                with st.expander('User input', expanded=True):
                    lower = st.number_input("lower bound", min_value=0, step=10)
                    upper = st.number_input("upper bound", min_value=0, step=10)

                    if st.button('Execute'):  
                        ##### show results
                        outlier_removal_update(lower, upper)
                        if len(outliers) > 0:
                            ####### add to history tab
                            st.session_state.hist_holder.append(f'{len(outliers)} outliers removed from {column_name} using Input')
        else:
            with coltype_holder.container():
                st.empty()
            st.warning('Please select column')


    ############################################################################################################
    ## Distributions Analysis ##################################################################################
    if ops == "Distributions":                        

        ### function to plot distributions
        def histogram_plotter():
            fig = px.histogram(dataset[column_name], nbins=40)
            fig.update_layout(margin=dict(t=30, b=0, l=0, r=80), 
                                title_text='Distribution of {} '.format(str(column_name)), title_x=0.3)
            return st.plotly_chart(fig, use_container_width=True)

        with img:
            if dataset[column_name].dtype == "int64" or dataset[column_name].dtype == "float64":
                hist_holder = st.empty()
                with hist_holder.container():
                    histogram_plotter()
                    st.write(pd.DataFrame(dataset[column_name].describe(include='all')).transpose())
            else:
                hist_holder = st.empty()
                with hist_holder.container():
                    histogram_plotter()


        with repair_options:
            ### show statistics and repair for numeric columns
            if dataset[column_name].dtype == "int64" or dataset[column_name].dtype == "float64":
                st.write('Options')
                #### remove range using user inpute
                with st.expander('Remove range of values'):
                    lower = st.number_input("lower bound", min_value=0, step=10)
                    upper = st.number_input("upper bound", min_value=0, step=10)

                    if st.button('Execute'):
                        out_idx = dataset[(dataset[column_name] > lower) & (dataset[column_name] < upper)].index.tolist() 
                        if len(out_idx) > 0:
                            d_copy = dataset.copy(deep=True)
                            append_data(d_copy)

                            dataset.drop(out_idx, inplace=True)
                            dataset.reset_index(drop=True, inplace=True)
                            ####### add to history tab
                            st.session_state.hist_holder.append(f'Values between {lower} and {upper} removed from {column_name}')
                            with hist_holder.container():
                                histogram_plotter()
                                st.write('Summary statistics')
                                st.write(pd.DataFrame(dataset[column_name].describe(include='all')).transpose())
                        else:
                            st.warning(f'There are no values between {lower} and {upper} in {column_name}')
                
            else:
                st.write('Options')
                #### remove values of specific classes from column
                with st.expander('Remove value'):
                    class_name = st.text_input("input class (case sensitive)")

                    if st.button('Execute'):
                        out_idx_s = dataset[dataset[column_name] == class_name].index.tolist()

                        if len(out_idx_s) > 0:  
                            d_copy = dataset.copy(deep=True)
                            append_data(d_copy)

                            dataset.drop(out_idx_s, inplace=True)
                            dataset.reset_index(drop=True, inplace=True)
                            ####### add to history tab
                            st.session_state.hist_holder.append(f'All {class_name} entries removed from {column_name}')
                            with hist_holder.container():
                                histogram_plotter()
                                

                        else:
                            st.warning(f'No entries with name {class_name}')
                    
                #### replace values of specific classes from column
                with st.expander('Replace value'):
                    target_class = st.text_input("Target class (case sensitive)")
                    replace_class = st.text_input("Replace with (case sensitive)")

                    if st.button('Replace'):
                        out_idx_s = dataset[dataset[column_name] == target_class].index.tolist()

                        if len(out_idx_s) > 0:
                            if replace_class is not None:
                                d_copy = dataset.copy(deep=True)
                                append_data(d_copy)

                                dataset[column_name][out_idx_s] = replace_class
                                dataset.reset_index(drop=True, inplace=True)
                                ####### add to history tab
                                st.session_state.hist_holder.append(f'All {target_class} in {column_name} replaced with {replace_class}')
                                with hist_holder.container():
                                    histogram_plotter()
                            else:
                                st.warning('Replacement is missing')

                        else:
                            st.warning(f'No entries with name {class_name}')

    ############################################################################################################
    ## Others ##################################################################################
    if ops == "Other":

        ### Calculate range
        ### function to replace range with mean
        def range_mean(row):
            row = str(row)
            if '-' in set(row):
                num_before = ""
                num_after = ""
                dash_idx = row.index('-')
                for n in row[0:dash_idx]:
                    if n.isdigit():
                        num_before = num_before + n
                for n in row[dash_idx:len(row)]:
                    if n.isdigit():
                        num_after = num_after + n
                return np.average([int(num_before), int(num_after)])
            else:
                return(row)

        with img:
            st.write('Column')
            single_column = st.empty()

            def update_col():
                with single_column.container():
                    col = pd.DataFrame(dataset[column_name])
                    return st.write(col)
            update_col()

        with repair_options:
            st.write('Options')
            if st.button('Replace range with mean'):
                d_copy = dataset.copy(deep=True)
                append_data(d_copy)
                dataset[column_name] = dataset[column_name].apply(range_mean)
                st.session_state.hist_holder.append(f'All range values in {column_name} replaced with mean values')
                update_col()

            ### Split Column
            with st.expander('Split column'):
                split_by = st.text_input('Split by:')
                new_col1 = st.text_input('New column 1 name: ')
                new_col2 = st.text_input('New column 2 name: ')
                Keep_col = st.radio('Keep original column', ['Yes', 'No'])

                def col_position_adjustment():
                    dataset.insert((dataset.columns.get_loc(column_name))+1, new_col1, dataset.pop(new_col1))
                    dataset.insert((dataset.columns.get_loc(column_name))+2, new_col2, dataset.pop(new_col2))

                if st.button('Execute'):
                    if Keep_col == 'Yes':
                        d_copy = dataset.copy(deep=True)
                        append_data(d_copy)
                        dataset[[new_col1,new_col2]] = dataset[column_name].str.split(split_by, 1, expand=True)
                        with single_column.container():
                            cols = pd.DataFrame(dataset[[column_name, new_col1, new_col2]])
                            st.write(cols)
                            st.session_state.hist_holder.append(f'{column_name} split into {new_col1, new_col2}')
                            col_position_adjustment()
                            with col_holder.container():
                                column_list = list(dataset)
                                column_name = st.selectbox("Select column to analyse", column_list)
                                st.info('{} column type is: {}'.format(column_name, col_type))
                            with coltype_holder.container():
                                st.empty()
                        
                            
                    else:
                        d_copy = dataset.copy(deep=True)
                        append_data(d_copy)
                        dataset[[new_col1,new_col2]] = dataset[column_name].str.split(split_by, 1, expand=True)
                        with single_column.container():
                            cols = pd.DataFrame(dataset[[new_col1, new_col2]])
                            st.write(cols)
                            st.session_state.hist_holder.append(f'{column_name} split into {new_col1, new_col2}, {column_name} dropped')
                        col_position_adjustment()
                        dataset.drop(columns=[str(column_name)], axis=1, inplace=True)
                        dataset.reset_index(drop=True, inplace=True)

                        
                        with col_holder.container():
                                column_list = list(dataset)
                                column_name = st.selectbox("Select column to analyse", column_list)
                                st.info('{} column type is: {}'.format(column_name, col_type))
                        with coltype_holder.container():
                            st.empty()

    input_hist()




#####################################################################################################################################
#################################################### Multiple Column Analysis #######################################################
#####################################################################################################################################
if main_options == 'Multiple Column Analysis':
    history_button()
    undo_button()

    with header_section.container():
        st.subheader("Multiple Column Analysis")

        ## start progress bar
        prog = st.progress(0)

        multi_ops = st.selectbox("Select Analysis", 
                                ["Missing Values", "Duplicates", "Correlations", "Distributions", "Scatter Plots", "Anomaly Detection (Split outliers)", "Clustering"])

        select_cols_hold = st.empty()

        with select_cols_hold.container():
            select_cols = st.multiselect('Select Columns', dataset.columns, default=list(dataset.columns))

        if len(select_cols) > 0:

            ### end progress bar
            prog.progress(100)

            ###########################################################################################################
            ### Missing values #########################################################################################
            if multi_ops == "Missing Values":
                
                #### column to show missing values using matrix
                with img:
                    miss_holder = st.empty()
                    def miss_plot():
                        fig = px.imshow(dataset[select_cols].notnull(), color_continuous_scale=px.colors.sequential.Blues)
                        fig.update_layout(margin=dict(t=30, b=0, l=0, r=0), 
                                            title_text='Missing values in dataset', 
                                            title_x=0.45, coloraxis_showscale=True)
                        fig.update_coloraxes(colorbar_tickvals=[0,250], colorbar_ticktext=['Null', 'Not null'], colorbar_thickness=10,
                                            colorbar_nticks=2)
                        
                        return st.plotly_chart(fig, use_container_width=True)

                    ##### table of missing values
                    with miss_holder.container():    
                        miss_plot()
                        st.write('Total no of missing values in each column')
                        dat = pd.DataFrame(dataset[select_cols].isnull().sum()).transpose()
                        dat

                    ##### button to drop all missing values
                    with repair_options:
                        st.write('Options')
                        if dataset[select_cols].isnull().sum().sum() != 0:
                            if st.button('Drop all null values'):

                                d_copy = dataset.copy(deep=True)
                                append_data(d_copy)

                                dataset.dropna(subset=select_cols, inplace=True)
                                dataset.reset_index(drop=True, inplace=True)
                
                                with miss_holder.container():
                                    st.success('All missing values removed successfully')
                                    miss_plot()
                                    st.write('Total no of missing values in each column')
                                    dat = pd.DataFrame(dataset[select_cols].isnull().sum()).transpose()
                                    dat
                                ####### add to history tab
                                st.session_state.hist_holder.append(f'Dropped all null values in {len(select_cols)} columns')
                            
                            st.warning('Consider single column analysis for null values instead')
                                
                        else:
                            st.info('There are no missing values in the selected columns')
                        
                        ###### remove selected columns
                        with st.expander('Remove selected columns'):
                            del_columns = st. multiselect('Remove multiple columns', dataset.columns)
                            if len(del_columns) > 0:
                                if st.button("Remove columns"):

                                    d_copy = dataset.copy(deep=True)
                                    append_data(d_copy)

                                    dataset.drop(columns=del_columns, inplace=True)
                                    #dataset.reset_index(drop=True, inplace=True)
                                    st.success('{} successfully removed'.format(del_columns))
                                    with select_cols_hold.container():
                                        select_cols = st.multiselect('Select columns', dataset.columns, default=list(dataset.columns))
                                    ####### add to history tab
                                    st.session_state.hist_holder.append(f'{del_columns} removed')
                                    with miss_holder.container():
                                        miss_plot()
                                        st.write('Total no of missing values in each column')
                                        dat = pd.DataFrame(dataset[select_cols].isnull().sum()).transpose()
                                        dat

                            else:
                                st.info('Please select columns')


            ############################################################################################################
            ### Duplicate Analysis ######################################################################################
            if multi_ops == 'Duplicates':
                if dataset[select_cols].isnull().sum().sum() > 0:
                    st.warning('Selected columns contain missing values which could lead to misinterpretation')

                #### check duplicates 
                def dup_df():
                    '''Check if duplicates exist and drop'''
                    dup_ent = dataset[select_cols][dataset[select_cols].duplicated()]
                    with img:
                        dup_holder = st.empty()

                    if len(dup_ent) > 0:
                        
                        with dup_holder.container():
                            st.info(f'There are {len(dup_ent)} duplicate enteries in the selected columns')
                            st.write(dup_ent)
                        
                        ###### drop duplicate entries
                        with repair_options:
                            st.write('Options')
                            if st.button('Drop duplicates'):

                                d_copy = dataset.copy(deep=True)
                                append_data(d_copy)

                                dataset.drop_duplicates(subset=select_cols,inplace=True)
                                dataset.reset_index(drop=True, inplace=True)
                                with dup_holder.container():
                                    st.empty()
                                    st.success('Duplicate rows removed successfully')
                                
                                ####### add to history tab
                                st.session_state.hist_holder.append(f' Duplicate rows removed from {len(select_cols)} columns')
                            
                    else:
                        st.success('There are no duplicate entries in the selected columns')

                dup_df()

            ###########################################################################################################
            ### Anomaly Detection ######################################################################################
            if multi_ops == "Anomaly Detection (Split outliers)":
                with select_cols_hold.container():
                    st.empty()
                
                #### selection columns for numeric and categorical data
                with repair_options:
                    num_data = st.selectbox('Select Y values (numeric)', dataset.select_dtypes(include=[np.number]).columns)
                    cat_data = st.selectbox('Select X values (categorical)', dataset.select_dtypes(exclude=[np.number]).columns)

                if len(dataset.select_dtypes(include=[np.number]).columns) > 0 and len(dataset.select_dtypes(exclude=[np.number]).columns) > 0:

                    #### boxplot of outliers
                    with img:
                        def multi_box_plot():
                            fig = px.box(dataset, x=str(cat_data), y=str(num_data))
                            fig.update_layout(margin=dict(t=30, b=0, l=0, r=20), 
                                                title_text=f'Boxplot of {num_data} versus {cat_data}', 
                                                title_x=0.5)

                            return st.plotly_chart(fig, use_container_width=True)

                        multi_box_plot()
                else:
                    st.warning('Please select columns')

            
            ###########################################################################################################
            ### Distributions #########################################################################################
            if multi_ops == "Distributions":

                if dataset.isna().sum().sum() > 0:
                    st.warning('Dataset contains missing values, this may lead to errors in plots')
                with select_cols_hold.container():
                    st.empty()
                
                #### get x and y data for plot
                #hist_col, options = st.columns([2,1])
                with repair_options:
                    y_data = st.selectbox('Select Y values (numeric)', dataset.select_dtypes(include=[np.number]).columns)
                    x_data = st.selectbox('Select X values (categorical)', dataset.select_dtypes(exclude=[np.number]).columns)
                    #mean_data = dataset.groupby([x_data]).mean().reset_index()
                    color_by = [None] + list(dataset.columns)
                    bar_col = st.selectbox('Select column to colour plot by', color_by)
                    
                #### make bar plot
                with img:
                    #@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
                    def multi_hist_plotter():
                        fig = px.histogram(data_frame=dataset, x=x_data, y=y_data, color=bar_col)
                        fig.update_layout(margin=dict(t=30, b=0, l=0, r=20), yaxis_title=f"Sum of {y_data}",
                                    title_text=f'Distribution of {y_data} by {x_data}', title_x=0.3)
                        return st.plotly_chart(fig, use_container_width=True)

                    multi_hist_plotter()

            ###########################################################################################################
            ### Scatter plots #########################################################################################
            if multi_ops == "Scatter Plots":
                if dataset.isna().sum().sum() > 0:
                    st.warning('Dataset contains missing values, this may lead to errors in plots')
                with select_cols_hold.container():
                    st.empty()
                
                #### get x and y data for scatter 
                #scatter, scatter_opt = st.columns([2,1])
                with repair_options:
                    y_dat = st.selectbox('Select Y values', dataset.select_dtypes(include=[np.number]).columns)
                    x_dat = st.selectbox('Select X values', dataset.select_dtypes(include=[np.number]).columns, index=1)
                    
                    color_by = [None] + list(dataset.columns)
                    bar_colscat = st.selectbox('Select column to colour plot', color_by)

                #### make sactter plot 
                with img:
                    #@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
                    def scatter_plotter():
                        fig = px.scatter(dataset, x=x_dat, y=y_dat, color=bar_colscat, color_continuous_scale=px.colors.sequential.Blues)
                        fig.update_layout(margin=dict(t=30, b=0, l=0, r=20), 
                                    title_text=f'Scatter plot of {y_dat} by {x_dat}', title_x=0.3)

                        return st.plotly_chart(fig, use_container_width=True)

                    scatter_plotter()
                    

            ###########################################################################################################
            ## Clustering ### from https://www.kaggle.com/code/dhanyajothimani/basic-visualization-and-clustering-in-python/notebook ########
            if multi_ops == "Clustering":
                if dataset.isna().sum().sum() > 0:
                    st.warning('Dataset contains missing values, please handle them to activate clustering')

                else:                
                    with select_cols_hold.container():
                        st.empty()
                    
                    f_list = st.multiselect('Select Columns (Numeric only)', dataset.select_dtypes(include=[np.number]).columns, 
                                            default=list(dataset.select_dtypes(include=[np.number]).columns))

                    if len(f_list) > 0:

                        #### get x and y data for scatter 
                        with repair_options:
                            st.header('')
                            st.header('')

                            f_dat = dataset[f_list]

                            ss = StandardScaler()
                            ss.fit_transform(np.array(f_dat).reshape(-1,1))

                            @st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
                            def n_clusters_finder(dat):
                                '''determine optimum number of clusters'''
                                
                                clust_range = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
                                sil_avg = []

                                for n_clusters in clust_range:
                                    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
                                    cluster_labels = clusterer.fit_predict(dat)
                                    silhouette_avg = silhouette_score(dat, cluster_labels)
                                    sil_avg.append(silhouette_avg)

                                return clust_range[np.argmax(sil_avg)]

                            nclust = n_clusters_finder(f_dat)

                            @st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
                            def cluster(dat, nclust):
                                model = KMeans(nclust)
                                model.fit(dat)
                                clust_labels = model.predict(dat)
                                return (clust_labels)

                            clust_labels = cluster(f_dat, nclust)
                            kmeans = pd.DataFrame(clust_labels)
                            f_dat.insert((f_dat.shape[1]),'agglomerative',kmeans)
                            
                            x_dat = st.selectbox('Select X values', f_list)
                            y_dat = st.selectbox('Select Y values', f_list)

                        with img:
                            st.info(f'Optimal number of clusters obtained from silhouette method is {nclust}')

                            def clust_plot():

                                fig = px.scatter(f_dat, x=x_dat, y=y_dat, color=kmeans[0])
                                fig.update_layout(margin=dict(t=30, b=0, l=0, r=20), 
                                            title_text=f'Scatter plot of {y_dat} by {x_dat}', title_x=0.3)

                                return st.plotly_chart(fig, use_container_width=True)
                            
                            clust_plot()
                    else:
                        st.warning('Please select columns')


            ###########################################################################################################
            ## Correlations ###########################################################################################
            if multi_ops == "Correlations":
                #st.info('Please refer to Data Summary section for Correlations')

                #### get numeric columns 
                num_data = dataset.select_dtypes(include=[np.number])

                with select_cols_hold.container():
                    select_cols = st.multiselect('Select Columns (Numeric only)', num_data.columns, default=list(num_data.columns))
                    st.info('Showing numeric columns only, see data profile for plot including categorical columns')

                    corr = dataset[select_cols].corr().round(3)
                
                with repair_options:
                    st.write('Options')
                    list_num = [None] + list(dataset[select_cols].columns)
                    cor_select = st.selectbox('Select target column', list_num)

                if cor_select is None:
                    with img:
                        ##### make heatmap corrplot of numeric columns
                        if len(select_cols) > 0:
                            def corr_plot():
                                fig = ff.create_annotated_heatmap(z=corr.to_numpy(), 
                                                x=corr.index.tolist(), 
                                                y=corr.columns.tolist(), 
                                                colorscale=px.colors.diverging.RdBu,
                                                zmin=-1,zmax=1,
                                                showscale=True,
                                                font_colors=['black']
                                                )

                                fig.update_layout(margin=dict(t=30, b=0, l=0, r=10), 
                                                title_text='Correlation plot', 
                                                title_x=0.5)
                                fig.update_xaxes(side="bottom")

                                return st.plotly_chart(fig, use_container_width=True)

                            corr_plot()
                        else:
                            st.warning('Please select columns')
                else:
                    with img:
                        if len(select_cols) > 0:
                            def cor_spec_plot():
                                fig = plt.figure(figsize=(12,12))
                                plt.rcParams["font.size"] = "20"
                                sns.heatmap(dataset[select_cols].corr()[[cor_select]].sort_values(by=cor_select,  ascending=False), 
                                vmin=-1, vmax=1, annot=True, cmap='RdBu')

                                plt.title(f'Correlation of all features with {cor_select}')
                                
                                return st.pyplot(fig)
                            cor_spec_plot()

        else:
            st.warning('Please select columns')

    input_hist()


####################################################################################################################################
# Download dataset after processing
data = dataset.to_csv(index=False).encode('utf-8')
st.sidebar.header('')
st.sidebar.header('')

st.sidebar.download_button(
    label="Download data as CSV ??????",
    data=data,
    file_name='cleaned_data.csv',
    mime='text/csv',
)