import streamlit as st
import itertools
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy import stats
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib
import seaborn as sns
from sklearn.pipeline import Pipeline
from wordcloud import WordCloud
from sklearn.decomposition import PCA
from scipy.stats import chi2_contingency, chi2
import statsmodels.api as sm
from scipy.stats import spearmanr

from scipy.stats import anderson

from PIL import Image
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import codecs

image = Image.open('Pulse Logo.png')
matplotlib.use("Agg")
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)

class DataFrame_Loader():

    def __init__(self):
        print("Loading DataFrame")

    def read_csv(self, data):
        self.df = pd.read_csv(data)
        return self.df


class EDA_Dataframe_Analysis():

    def __init__(self):

        print("General_EDA object created")

    def show_dtypes(self, x):
        return x.dtypes

    def show_columns(self, x):
        return x.columns

    def Show_Missing(self, x):
        return x.isna().sum()

    def Show_Missing1(self, x):
        return x.isna().sum()

    def Show_Missing2(self, x):
        return x.isna().sum()

    def show_hist(self, x):
        return x.hist()

    def Tabulation(self, x):
        table = pd.DataFrame(x.dtypes, columns=['dtypes'])
        table1 = pd.DataFrame(x.columns, columns=['Names'])
        table = table.reset_index()
        table = table.rename(columns={'index': 'Name'})
        table['No of Missing'] = x.isnull().sum().values
        table['No of Uniques'] = x.nunique().values
        table['Percent of Missing'] = ((x.isnull().sum().values) / (x.shape[0])) * 100
        table['First Observation'] = x.loc[0].values
        table['Second Observation'] = x.loc[1].values
        table['Third Observation'] = x.loc[2].values
        for name in table['Name'].value_counts().index:
            table.loc[table['Name'] == name, 'Entropy'] = round(
                stats.entropy(x[name].value_counts(normalize=True), base=2), 2)
        return table

    def Numerical_variables(self, x):
        Num_var = [var for var in x.columns if x[var].dtypes != "object"]
        Num_var = x[Num_var]
        return Num_var

    def categorical_variables(self, x):
        cat_var = [var for var in x.columns if x[var].dtypes == "object"]
        cat_var = x[cat_var]
        return cat_var

    def impute(self, x):
        df = x.dropna()
        return df

    def imputee(self, x):
        df = x.dropna()
        return df

    def Show_pearsonr(self, x, y):
        result = pearsonr(x, y)
        return result

    def Show_spearmanr(self, x, y):
        result = spearmanr(x, y)
        return result

    def plotly(self, a, x, y):
        fig = px.scatter(a, x=x, y=y)
        fig.update_traces(marker=dict(size=10,
                                      line=dict(width=2,
                                                color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        fig.show()

    def show_displot(self, x):
        plt.figure(1)
        plt.subplot(121)
        sns.distplot(x)

        plt.subplot(122)
        x.plot.box(figsize=(16, 5))

        plt.show()

    def Show_DisPlot(self, x):
        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(12, 7))
        return sns.distplot(x, bins=25)

    def Show_CountPlot(self, x):
        fig_dims = (18, 8)
        fig, ax = plt.subplots(figsize=fig_dims)
        return sns.countplot(x, ax=ax)

    def plotly_histogram(self, a, x, y):
        fig = px.histogram(a, x=x, y=y)
        fig.update_traces(marker=dict(size=10,
                                      line=dict(width=2,
                                                color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        fig.show()

    def plotly_violin(self, a, x, y):
        fig = px.histogram(a, x=x, y=y)
        fig.update_traces(marker=dict(size=10,
                                      line=dict(width=2,
                                                color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        fig.show()

    def Show_PairPlot(self, x):
        return sns.pairplot(x)

    def Show_HeatMap(self, x):
        f, ax = plt.subplots(figsize=(15, 15))
        return sns.heatmap(x.corr(), annot=True, ax=ax);

    def wordcloud(self, x):
        wordcloud = WordCloud(width=1000, height=500).generate(" ".join(x))
        plt.imshow(wordcloud)
        plt.axis("off")
        return wordcloud

    def label(self, x):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        x = le.fit_transform(x)
        return x

    def label1(self, x):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        x = le.fit_transform(x)
        return x

    def concat(self, x, y, z, axis):
        return pd.concat([x, y, z], axis)

    def dummy(self, x):
        return pd.get_dummies(x)

    def qqplot(self, x):
        return sm.qqplot(x, line='45')

    def Anderson_test(self, a):
        return anderson(a)

    def PCA(self, x):
        pca = PCA(n_components=8)
        principlecomponents = pca.fit_transform(x)
        principledf = pd.DataFrame(data=principlecomponents)
        return principledf

    def outlier(self, x):
        high = 0
        q1 = x.quantile(.25)
        q3 = x.quantile(.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high += q3 + 1.5 * iqr
        outlier = (x.loc[(x < low) | (x > high)])
        return (outlier)

    def check_cat_relation(self, x, y, confidence_interval):
        cross_table = pd.crosstab(x, y, margins=True)
        stat, p, dof, expected = chi2_contingency(cross_table)
        print("Chi_Square Value = {0}".format(stat))
        print("P-Value = {0}".format(p))
        alpha = 1 - confidence_interval
        return p, alpha
        if p > alpha:
            print(">> Accepting Null Hypothesis <<")
            print("There Is No Relationship Between Two Variables")
        else:
            print(">> Rejecting Null Hypothesis <<")
            print("There Is A Significance Relationship Between Two Variables")


class Attribute_Information():

    def __init__(self):

        print("Attribute Information object created")

    def Column_information(self, data):

        data_info = pd.DataFrame(
            columns=['No of observation',
                     'No of Variables',
                     'No of Numerical Variables',
                     'No of Factor Variables',
                     'No of Categorical Variables',
                     'No of Logical Variables',
                     'No of Date Variables',
                     'No of zero variance variables'])

        data_info.loc[0, 'No of observation'] = data.shape[0]
        data_info.loc[0, 'No of Variables'] = data.shape[1]
        data_info.loc[0, 'No of Numerical Variables'] = data._get_numeric_data().shape[1]
        data_info.loc[0, 'No of Factor Variables'] = data.select_dtypes(include='category').shape[1]
        data_info.loc[0, 'No of Logical Variables'] = data.select_dtypes(include='bool').shape[1]
        data_info.loc[0, 'No of Categorical Variables'] = data.select_dtypes(include='object').shape[1]
        data_info.loc[0, 'No of Date Variables'] = data.select_dtypes(include='datetime64').shape[1]
        data_info.loc[0, 'No of zero variance variables'] = data.loc[:, data.apply(pd.Series.nunique) == 1].shape[1]

        data_info = data_info.transpose()
        data_info.columns = ['value']
        data_info['value'] = data_info['value'].astype(int)

        return data_info

    def __get_missing_values(self, data):

        # Getting sum of missing values for each feature
        missing_values = data.isnull().sum()
        # Feature missing values are sorted from few to many
        missing_values.sort_values(ascending=False, inplace=True)

        # Returning missing values
        return missing_values

    def __iqr(self, x):
        return x.quantile(q=0.75) - x.quantile(q=0.25)

    def __outlier_count(self, x):
        upper_out = x.quantile(q=0.75) + 1.5 * self.__iqr(x)
        lower_out = x.quantile(q=0.25) - 1.5 * self.__iqr(x)
        return len(x[x > upper_out]) + len(x[x < lower_out])

    def num_count_summary(self, df):
        df_num = df._get_numeric_data()
        data_info_num = pd.DataFrame()
        i = 0
        for c in df_num.columns:
            data_info_num.loc[c, 'Negative values count'] = df_num[df_num[c] < 0].shape[0]
            data_info_num.loc[c, 'Positive values count'] = df_num[df_num[c] > 0].shape[0]
            data_info_num.loc[c, 'Zero count'] = df_num[df_num[c] == 0].shape[0]
            data_info_num.loc[c, 'Unique count'] = len(df_num[c].unique())
            data_info_num.loc[c, 'Negative Infinity count'] = df_num[df_num[c] == -np.inf].shape[0]
            data_info_num.loc[c, 'Positive Infinity count'] = df_num[df_num[c] == np.inf].shape[0]
            data_info_num.loc[c, 'Missing Percentage'] = df_num[df_num[c].isnull()].shape[0] / df_num.shape[0]
            data_info_num.loc[c, 'Count of outliers'] = self.__outlier_count(df_num[c])
            i = i + 1
        return data_info_num

    def statistical_summary(self, df):

        df_num = df._get_numeric_data()

        data_stat_num = pd.DataFrame()

        try:
            data_stat_num = pd.concat([df_num.describe().transpose(),
                                       pd.DataFrame(df_num.quantile(q=0.10)),
                                       pd.DataFrame(df_num.quantile(q=0.90)),
                                       pd.DataFrame(df_num.quantile(q=0.95))], axis=1)
            data_stat_num.columns = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', '10%', '90%', '95%']
        except:
            pass

        return data_stat_num


# def get_table_download_link(df):
#     """Generates a link allowing the data in a given panda dataframe to be downloaded
#     in:  dataframe
#     out: href string
#     """
#     csv = df.to_csv(index=False)
#     #b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
#     href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'

import os
import base64
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

st.image(image, use_column_width=True)


def main():


    st.title("Pulse Exploratory Data Analytics")

    st.info("Upload an Engage .csv or any .csv file at all! ")
    st.info("Download Link for PandasProfiling interactive HTML file will generate below ")
    #"""https://github.com/joeperrotta/streamlit"""
    activities = ["PandasProfiling", "General EDA", ]
    choice = st.sidebar.selectbox("Select EDA App", activities)

    if choice == "PandasProfiling":
        st.markdown(get_binary_file_downloader_html('IST Research EDA.html', 'Interactive HTML'),
                    unsafe_allow_html=True)
        st.subheader("Automated EDA with PandasProfiling")
        data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])


        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
            profile = ProfileReport(df)
            st_profile_report(profile)
            ### change settings to run no correlations if error occurs, probably from free text fields
            # r = ProfileReport(samples=None, correlations=None, missing_diagrams=None, duplicates=None, interactions=None)
            #### Report Title
            profile = df.profile_report(title='IST Research EDA')
            #### Report File
            profile.to_file(output_file='IST Research EDA.html')
            st.markdown(get_binary_file_downloader_html('IST Research EDA.html', 'Interactive HTML'), unsafe_allow_html=True)


    elif choice == 'General EDA':
        st.subheader("Exploratory Data Analysis")

        data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
        if data is not None:
            df = load.read_csv(data)
            st.dataframe(df.head())
            st.success("Data Frame Loaded successfully")

            if st.checkbox("Show dtypes"):
                st.write(dataframe.show_dtypes(df))

            if st.checkbox("Show Columns"):
                st.write(dataframe.show_columns(df))

            if st.checkbox("Show Missing"):
                st.write(dataframe.Show_Missing1(df))

            if st.checkbox("column information"):
                st.write(info.Column_information(df))

            if st.checkbox("Aggregation Tabulation"):
                st.write(dataframe.Tabulation(df))

            if st.checkbox("Num Count Summary"):
                st.write(info.num_count_summary(df))

            if st.checkbox("Statistical Summary"):
                st.write(info.statistical_summary(df))

            # 			if st.checkbox("Show Selected Columns"):
            # 				selected_columns = st.multiselect("Select Columns",all_columns)
            # 				new_df = df[selected_columns]
            # 				st.dataframe(new_df)

            if st.checkbox("Show Selected Columns"):
                selected_columns = st.multiselect("Select Columns", dataframe.show_columns(df))
                new_df = df[selected_columns]
                st.dataframe(new_df)

            if st.checkbox("Numerical Variables"):
                num_df = dataframe.Numerical_variables(df)
                numer_df = pd.DataFrame(num_df)
                st.dataframe(numer_df)

            if st.checkbox("Categorical Variables"):
                new_df = dataframe.categorical_variables(df)
                catego_df = pd.DataFrame(new_df)
                st.dataframe(catego_df)

            if st.checkbox("DropNA"):
                imp_df = dataframe.impute(num_df)
                st.dataframe(imp_df)

            if st.checkbox("Missing after DropNA"):
                st.write(dataframe.Show_Missing(imp_df))

            all_columns_names = dataframe.show_columns(df)
            all_columns_names1 = dataframe.show_columns(df)
            selected_columns_names = st.selectbox("Select Column 1 For Cross Tabultion", all_columns_names)
            selected_columns_names1 = st.selectbox("Select Column 2 For Cross Tabultion", all_columns_names1)
            if st.button("Generate Cross Tab"):
                st.dataframe(pd.crosstab(df[selected_columns_names], df[selected_columns_names1]))

            all_columns_names3 = dataframe.show_columns(df)
            all_columns_names4 = dataframe.show_columns(df)
            selected_columns_name3 = st.selectbox("Select Column 1 For Pearsonr Correlation (Numerical Columns)",
                                                  all_columns_names3)
            selected_columns_names4 = st.selectbox("Select Column 2 For Pearsonr Correlation (Numerical Columns)",
                                                   all_columns_names4)
            if st.button("Generate Pearsonr Correlation"):
                df = pd.DataFrame(
                    dataframe.Show_pearsonr(imp_df[selected_columns_name3], imp_df[selected_columns_names4]),
                    index=['Pvalue', '0'])
                st.dataframe(df)

            spearmanr3 = dataframe.show_columns(df)
            spearmanr4 = dataframe.show_columns(df)
            spearmanr13 = st.selectbox("Select Column 1 For spearmanr Correlation (Categorical Columns)", spearmanr4)
            spearmanr14 = st.selectbox("Select Column 2 For spearmanr Correlation (Categorical Columns)", spearmanr4)
            if st.button("Generate spearmanr Correlation"):
                df = pd.DataFrame(dataframe.Show_spearmanr(catego_df[spearmanr13], catego_df[spearmanr14]),
                                  index=['Pvalue', '0'])
                st.dataframe(df)

            st.subheader("UNIVARIATE ANALYSIS")

            all_columns_names = dataframe.show_columns(df)
            selected_columns_names = st.selectbox("Select Column for Histogram ", all_columns_names)
            if st.checkbox("Show Histogram for Selected variable"):
                st.write(dataframe.show_hist(df[selected_columns_names]))
                st.pyplot()

            all_columns_names = dataframe.show_columns(df)
            selected_columns_names = st.selectbox("Select Columns Distplot ", all_columns_names)
            if st.checkbox("Show DisPlot for Selected variable"):
                st.write(dataframe.Show_DisPlot(df[selected_columns_names]))
                st.pyplot()

            all_columns_names = dataframe.show_columns(df)
            selected_columns_names = st.selectbox("Select Columns CountPlot ", all_columns_names)
            if st.checkbox("Show CountPlot for Selected variable"):
                st.write(dataframe.Show_CountPlot(df[selected_columns_names]))
                st.pyplot()

            st.subheader("BIVARIATE ANALYSIS")

            Scatter1 = dataframe.show_columns(df)
            Scatter2 = dataframe.show_columns(df)
            Scatter11 = st.selectbox("Select Column 1 For Scatter Plot (Numerical Columns)", Scatter1)
            Scatter22 = st.selectbox("Select Column 2 For Scatter Plot (Numerical Columns)", Scatter2)
            if st.button("Generate PLOTLY Scatter PLOT"):
                st.pyplot(dataframe.plotly(df, df[Scatter11], df[Scatter22]))

            bar1 = dataframe.show_columns(df)
            bar2 = dataframe.show_columns(df)
            bar11 = st.selectbox("Select Column 1 For Bar Plot ", bar1)
            bar22 = st.selectbox("Select Column 2 For Bar Plot ", bar2)
            if st.button("Generate PLOTLY histogram PLOT"):
                st.pyplot(dataframe.plotly_histogram(df, df[bar11], df[bar22]))

            violin1 = dataframe.show_columns(df)
            violin2 = dataframe.show_columns(df)
            violin11 = st.selectbox("Select Column 1 For violin Plot", violin1)
            violin22 = st.selectbox("Select Column 2 For violin Plot", violin2)
            if st.button("Generate PLOTLY violin PLOT"):
                st.pyplot(dataframe.plotly_violin(df, df[violin11], df[violin22]))

            st.subheader("MULTIVARIATE ANALYSIS")

            if st.checkbox("Show Histogram"):
                st.write(dataframe.show_hist(df))
                st.pyplot()

            if st.checkbox("Show HeatMap"):
                st.write(dataframe.Show_HeatMap(df))
                st.pyplot()

            if st.checkbox("Show PairPlot"):
                st.write(dataframe.Show_PairPlot(df))
                st.pyplot()

            if st.button("Generate Word Cloud"):
                st.write(dataframe.wordcloud(df))
                st.pyplot()





    st.markdown('Powered by **_Pulse_**')
    st.markdown('<style>h1{color: blue;}</style>', unsafe_allow_html=True)
    # st.title("Credits and Inspiration")
    # """https://pycaret.org/"""


if __name__ == '__main__':
    load = DataFrame_Loader()
    dataframe = EDA_Dataframe_Analysis()
    info = Attribute_Information()
    main()
