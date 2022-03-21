from pycaret.anomaly import *
import streamlit as st
from streamlit import caching
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from collections import Counter
import os
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
import missingno as msno
import seaborn as sns
import plotly.figure_factory as ff


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
main_options = st.sidebar.radio("Select Task", ["Data Summary", "Single Column Analysis", "Multiple Column Analysis"])


########################################################################################
# DQ Summary #################
if main_options == "Data Summary":

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
if main_options == 'Single Column Analysis':
    st.subheader("Single Column Analysis")

    ## start progress bar
    prog = st.progress(0) 

    ## Dropdown list of type of analysis
    #Single_ops = 
    ops = st.selectbox("Type of analysis", ["Missing Values", "Outliers", "Entry Type", "Distributions"])

    ## Container to hold dropdown list of each column
    col_holder = st.empty()

    ## Insert column list into container
    with col_holder.container():
        column_list = list(dataset)
        column_name = st.selectbox("Select column to analyse", column_list)

    ## show column type
    coltype_holder = st.empty()
    def update_coltype():
        with coltype_holder.container():
            data_conv = dataset.convert_dtypes()
            col_type = data_conv[column_name].dtype
            st.info('{} column type is: {}'.format(column_name, col_type))

    update_coltype()

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

            colors = ['red', 'green']
            fig.update_traces(textfont_size=17,
                  marker=dict(colors=colors))
            fig.update_layout(margin=dict(t=30, b=0, l=0, r=80), title_text='Null values in {} '.format(str(column_name)), title_x=0.3)
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

            def recompute_and_plot():
                '''recompute missing values and update plot'''
                with pie_holder.container():
                            missing_df = compute_missing(dataset, column_name)
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
                    dataset.reset_index(drop=True, inplace=True)
                
                    recompute_and_plot()
                    st.success('Null values successfully removed')

                #### replace missing values with mean  ############################################  
                if st.button("Replace null values with mean"):
                    if dataset[column_name].dtype == "int64" or dataset[column_name].dtype == "float64":
                        #missing_index = dataset[dataset[column_name].isnull()].index.tolist()
                        dataset[column_name].fillna(dataset[column_name].mean(), inplace=True)

                        recompute_and_plot()
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
                        imputer = KNNImputer(n_neighbors=5)
                        df_new = pd.DataFrame(imputer.fit_transform(df_int),columns = df_int.columns)

                        df_new[int_data.columns] = scaler.inverse_transform(df_new[int_data.columns])

                        dataset[int_data.columns] = df_new[int_data.columns]

                        recompute_and_plot()
                        st.success('Null values successfully replaced with nearest neighbours')


                    else:
                        st.warning("Numeric column only")

                #### repair missing value using user inpute ################################################
                with st.expander('Replace with specified value'):
                    if dataset[column_name].dtype == "int64" or dataset[column_name].dtype == "float64":
                        user_miss_input = st.number_input("Input", min_value=0)
                
                    else:
                        user_miss_input = st.text_input("Input")

                    if st.button('Replace'):
                        dataset[column_name].fillna(value=user_miss_input, inplace=True)
                        ##### show results
                        recompute_and_plot()
                        st.success('Null values successfully replaced with user iput')

                #### remove column ##############################################
                if st.button("Remove column"):
                    dataset.drop(columns=[str(column_name)], axis=1, inplace=True)
                    st.success('{} successfully removed'.format(str(column_name)))

                    with col_holder.container():
                        column_list = list(dataset)
                        column_name = st.selectbox("Select column to analyse", column_list)

                    recompute_and_plot()
                                                
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

                    recompute_and_plot()

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
        def compute_datatype():
            dataset.reset_index(drop=True, inplace=True)
            #'''This function determines datatypes and indexes'''
            global uniq_list_dtypes, uniq_counts, list_dtypes, digit_index, str_index, other_index

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
            #compute_datatype()
           
            with bar_holder.container():
                compute_datatype()
                barplotter(uniq_list_dtypes, uniq_counts)
        
        ### column holding remedy buttons 
        with rem:
            compute_datatype()
            st.subheader('')
            st.write('Perform repair action')

            if len(uniq_list_dtypes) > 1:

                #### delete numeric entries ###############################################################
                if st.button('Delete numeric entries'):
                    if len(digit_index) > 0:
                        dataset.drop(digit_index, inplace=True)
                        compute_datatype()
                        st.success('numeric enteries removed')
                        with bar_holder.container():
                            barplotter(uniq_list_dtypes, uniq_counts)
                    else:
                        st.info('No numeric enteries found')

                #### delete string entries ################################################################
                if st.button('Delete string entries'):
                    if len(str_index) > 0:
                        dataset.drop(str_index, inplace=True)
                        compute_datatype()
                        st.success('string entries removed')
                        with bar_holder.container():
                            barplotter(uniq_list_dtypes, uniq_counts)
                    else:
                        st.info('No string enteries found')

                #### delete other entries #################################################################
                if st.button('Delete other entries'):
                    if len(other_index) >0:
                        dataset.drop(other_index, inplace=True)
                        compute_datatype()
                        st.success('other entries removed')
                        with bar_holder.container():
                            barplotter(uniq_list_dtypes, uniq_counts) 
                    else:
                        st.info('No other entries found')

            elif len(uniq_list_dtypes) == 1:
                st.info('Column has one entry type')

            else: 
                st.error('Unable to identify type of entries in this column')


    ################################################################################################
    ## Outlier Ananlysis############################################################################
    if ops == "Outliers": # codes from https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/ and 
                           # https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
        
        st.subheader("Outlier identification and removal")

        with col_holder.container():
            column_list = dataset.select_dtypes(include=[np.number]).columns
            column_name = st.selectbox("Select column to analyse", column_list)
            st.info('Showing numeric columns only')

        ### update coltype_holder
        update_coltype()

        ### columns to hold plot and repair methods
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
                    st.success('{} outliers removed successfully'.format(len(outliers)))

                else:
                    st.info('No identified outliers')

            #### show outliers in df, remove outliers and update boxplot
            def outlier_removal_update(lower, upper):
                    #show outliers
                    show_outlier_df(lower, upper)
                    ##### identify and remove outliers
                    out_idx = dataset[(dataset[column_name] < lower) | (dataset[column_name] > upper)].index.tolist()
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

            #### repair outliers using user inpute
            with st.expander('User input'):
                lower = st.number_input("lower bound", min_value=0, step=10)
                upper = st.number_input("upper bound", min_value=0, step=10)

                if st.button('Execute'):  
                    ##### show results
                    outlier_removal_update(lower, upper)


    ############################################################################################################
    ## Distributions Analysis ##################################################################################
    if ops == "Distributions":
        st.subheader('Column distribution')

        def hist_plotter():
            fig = px.histogram(dataset[column_name], nbins=40)
            fig.update_layout(margin=dict(t=30, b=0, l=0, r=80), 
                                title_text='Distribution of {} '.format(str(column_name)), title_x=0.3)
            return st.plotly_chart(fig, use_container_width=True)

        hist, stats = st.columns([2,1])

        with hist:
            hist_holder = st.empty()

            with hist.container():
                hist_plotter()

        if dataset[column_name].dtype == "int64" or dataset[column_name].dtype == "float64":
            with stats:
                st.write('Summary statistics')
                st.write(dataset[column_name].describe(include='all'))

#####################################################################################################################################
#################################################### Multiple Column Analysis #######################################################
#####################################################################################################################################
if main_options == 'Multiple Column Analysis':
    st.subheader("Multiple Column Analysis")

    ## start progress bar
    prog = st.progress(0)

    multi_ops = st.selectbox("Select Analysis", 
                            ["Missing values", "Clusters and Outliers", "Anomaly Detection", 
                            "Duplicates", "Correlations", "Scatter and Distributions"])

    select_cols_hold = st.empty()

    with select_cols_hold.container():
        select_cols = st.multiselect('Select columns', dataset.columns, default=list(dataset.columns))

    if len(select_cols) > 0:

        ## end progress bar
        prog.progress(100)

        ###########################################################################################################
        ## Missing values #########################################################################################
        if multi_ops == "Missing values":

            mat, repair = st.columns([2,1])
            
            with mat:
                miss_holder = st.empty()
                def miss_plot():
                    fig = px.imshow(dataset[select_cols].notnull(), color_continuous_scale=px.colors.sequential.Blues)
                    fig.update_layout(margin=dict(t=30, b=0, l=0, r=0), 
                                        title_text='Missing values in dataset', 
                                        title_x=0.45, coloraxis_showscale=False)
                    
                    return st.plotly_chart(fig, use_container_width=True)

                with miss_holder.container():    
                    miss_plot()
                    st.write('Total no of missing values in each column')
                    dat = pd.DataFrame(dataset[select_cols].isnull().sum()).transpose()
                    dat

                with repair:
                    if dataset[select_cols].isnull().sum().sum() != 0:
                        st.header('')
                        if st.button('Drop all missing values'):
                            dataset.dropna(subset=select_cols, inplace=True)
            
                            with miss_holder.container():
                                st.success('All missing values removed successfully')
                                miss_plot()
                                st.write('Total no of missing values in each column')
                                dat = pd.DataFrame(dataset[select_cols].isnull().sum()).transpose()
                                dat
                        
                        st.warning('Dropping all missing values in one go is not advised. Consider doing this through single column analysis instead')
                            
                        #### repair missing value using user inpute ################################################
                        with st.expander('Replace with specified value'):
                            if len(dataset[select_cols].dtypes[dataset[select_cols].dtypes == 'int64']
                                    [dataset[select_cols].dtypes == 'float64']) == len(dataset[select_cols].dtypes):
                                user_miss_input = st.number_input("Input")

                                if st.button('Replace'):
                                    dataset[select_cols].fillna(value=user_miss_input, inplace=True)
                                    ##### show results
                                    with miss_holder.container():
                                        st.success('Null values successfully replaced with user input')
                                        miss_plot()

                            elif len(dataset[select_cols].dtypes[dataset[select_cols].dtypes != 'int64']
                                    [dataset[select_cols].dtypes != 'float64']) == len(dataset[select_cols].dtypes):
                                user_miss_input = st.text_input("Input")

                                if st.button('Replace'):
                                    dataset[select_cols].fillna(value=user_miss_input, inplace=True)
                                    ##### show results
                                    with miss_holder.container():
                                        st.success('Null values successfully replaced with user iput')
                                        miss_plot()

                            
                    else:
                        st.info('There are no missing values in the selected columns')


        ############################################################################################################
        ## Duplicate Analysis ######################################################################################
        if multi_ops == 'Duplicates':

            dup, drop_dup = st.columns([2,1])
            def dup_df():
                '''Check if duplicates exist and drop'''
                dup_ent = dataset[select_cols][dataset[select_cols].duplicated()]
                with dup:
                    dup_holder = st.empty()

                if len(dup_ent) > 0:
                    
                    with dup_holder.container():
                        st.info(f'There are {len(dup_ent)} duplicate enteries in the selected columns')
                        st.write(dup_ent)
                    
                    with drop_dup:
                        if st.button('Drop duplicates'):
                            dataset.drop_duplicates(subset=select_cols,inplace=True)
                            with dup_holder.container():
                                st.empty()
                                st.success('Duplicate rows removed successfully')
                         
                else:
                    st.success('There are no duplicate entries in the selected columns')

            dup_df()

        ###########################################################################################################
        ## Anomaly Detection ######################################################################################
        if multi_ops == "Anomaly Detection":
            with select_cols_hold.container():
                st.empty()
            
            box_col, options = st.columns([2,1])
            with options:
                num_data = st.selectbox('Select Y values (numeric)', dataset.select_dtypes(include=[np.number]).columns)
                cat_data = st.selectbox('Select X values (categorical)', dataset.select_dtypes(exclude=[np.number]).columns)

            with box_col:

                def multi_box_plot():
                    fig = px.box(dataset, x=str(cat_data), y=str(num_data))
                    fig.update_layout(margin=dict(t=30, b=0, l=0, r=20), 
                                        title_text=f'Boxplot of {num_data} versus {cat_data}', 
                                        title_x=0.5)

                    return st.plotly_chart(fig, use_container_width=True)

                multi_box_plot()

        
        ###########################################################################################################
        ## Scatter and Distributions ######################################################################################
        if multi_ops == "Scatter and Distributions":

            if dataset.isna().sum().sum() > 0:
                st.warning('Dataset contains missing values, this may lead to errors in plots')
            with select_cols_hold.container():
                st.empty()
            
            hist_col, options = st.columns([2,1])
            with options:
                x_data = st.selectbox('Select X values ', dataset.columns)
                y_data = st.selectbox('Select Y values', dataset.columns)
                bar_col = st.selectbox('Select column to colour plot by', dataset.columns)
                

            with hist_col:

                def multi_hist_plotter():
                    fig = px.bar(dataset, x=x_data, y=y_data, color=bar_col)
                    fig.update_layout(margin=dict(t=30, b=0, l=0, r=20), 
                                title_text=f'Distribution of {x_data} by {y_data}', title_x=0.3)
                    return st.plotly_chart(fig, use_container_width=True)

                multi_hist_plotter()

            scatter, scatter_opt = st.columns([2,1])
            with scatter_opt:
                x_dat = st.selectbox('Select X values', dataset.select_dtypes(include=[np.number]).columns)
                y_dat = st.selectbox('Select Y values', dataset.select_dtypes(include=[np.number]).columns)
                bar_colscat = st.selectbox('Select column to colour plot', dataset.columns)

            with scatter:
                def scatter_plotter():
                    fig = px.scatter(dataset, x=x_dat, y=y_dat, color=bar_colscat)
                    fig.update_layout(margin=dict(t=30, b=0, l=0, r=20), 
                                title_text=f'Scatter plot of {y_dat} by {x_dat}', title_x=0.3)

                    return st.plotly_chart(fig, use_container_width=True)

                scatter_plotter()
                

        ###########################################################################################################
        ## Transpose Data #########################################################################################
        if multi_ops == "Transpose":
            st.write('Anomaly detection using PyCaret')

        ###########################################################################################################
        ## Correlations ###########################################################################################
        if multi_ops == "Correlations":
            #st.info('Please refer to Data Summary section for Correlations')

            num_data = dataset.select_dtypes(include=[np.number])

            with select_cols_hold.container():
                select_cols = st.multiselect('Select columns', num_data.columns, default=list(num_data.columns))
                st.info('Showing numeric columns only')

                corr = dataset[select_cols].corr().round(3)

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

                        #fig = go.Figure(data=[heat])
                        fig.update_layout(margin=dict(t=30, b=0, l=0, r=80), 
                                        title_text='Correlation plot', 
                                        title_x=0.5)
                        fig.update_xaxes(side="bottom")

                        return st.plotly_chart(fig, use_container_width=True)

                    corr_plot()
                else:
                    st.info('Please select columns')




    else:
        st.info('Please select columns')


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
