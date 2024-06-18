# Data Manipulation and Preprocessing
import numpy as np
import pandas as pd
import json

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
from plotnine import *

# TF-IDF and Dimensionality Reduction
from sklearn.decomposition import TruncatedSVD, PCA, NMF
from sklearn.feature_extraction.text import TfidfVectorizer

# Clustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster


##### Stopwords #####
# For removing the stopwords 
with open('stopwords_dmw.txt', 'r') as file:
    # Read the contents of the file
    stopwords_content = file.read()

# Split the contents into individual stopwords
stopwords_list = stopwords_content.split('\n')

# Optionally, remove any empty strings from the list
stopwords_list = [word for word in stopwords_list if word.strip()]


##### Functions #####
def visualize_sparsity(abstract_list, min_df_range=None, max_df_range=None):
    """
    Visualize the sparsity of TF-IDF matrices for different min_df and max_df values.

    Parameters
    ----------
    abstract_list : list
        List of abstracts from a specific time range.
    min_df_range : numpy.ndarray, optional
        Array containing the range of min_df values to test.
        Default is None, which will generate min_df values from 0.0 to 0.05.
    max_df_range : numpy.ndarray, optional
        Array containing the range of max_df values to test.
        Default is None, which will generate max_df values from 0.6 to 1.0.

    Returns
    -------
    None
        Displays the heatmap of sparsity.
    """
    if min_df_range is None:
        min_df_range = np.linspace(0.0, 0.05, num=11, endpoint=False)[::-1]
    if max_df_range is None:
        max_df_range = np.linspace(0.6, 1.0, num=5)

    # Create empty matrix to store results
    results_matrix = np.zeros((len(min_df_range), len(max_df_range)))

    # Iterate over min_df and max_df values and calculate sparsity
    for i, min_df in enumerate(min_df_range):
        for j, max_df in enumerate(max_df_range):
            vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df)
            tfidf_matrix = vectorizer.fit_transform(abstract_list)
            sparsity = 1.0 - np.count_nonzero(tfidf_matrix.toarray()) / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])
            results_matrix[i, j] = sparsity

    # Create heatmap
    plt.figure(figsize=(10, 6), dpi=250)
    sns.heatmap(results_matrix, annot=False, fmt=".3f", cmap="Reds",
                xticklabels=["{:.2f}".format(max_df) for max_df in max_df_range],
                yticklabels=["{:.2f}".format(min_df) for min_df in min_df_range])
    plt.xlabel('max_df')
    plt.ylabel('min_df')
    plt.title('Sparsity of TF-IDF Matrix for Different min_df and max_df Values')
    plt.show()
    
    
def vectorize_abstract(abstracts, min_df=0.03, max_df=0.8):
    """
    Vectorize text data using TF-IDF vectorization.

    Parameters
    ----------
    abstracts : list or pandas.Series
        The abstracts to be vectorized.
    stopwords_list : list, optional
        List of stopwords to be removed during vectorization.
        Default is None.
    min_df : float, optional
        The minimum document frequency. Words with a document frequency lower
        than this value will be ignored.
        Default is 0.03.
    max_df : float, optional
        The maximum document frequency. Words with a document frequency higher
        than this value will be ignored.
        Default is 0.8.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the TF-IDF matrix representation of the text data.
    """
    # Vectorize the data
    vectorizer = TfidfVectorizer(token_pattern=r"\b[a-z']+\b", lowercase=True,
                                 stop_words='english', min_df=min_df,
                                 max_df=max_df)
    tfidf_matrix = vectorizer.fit_transform(abstracts)

    # Convert TF-IDF matrix to DataFrame
    df_tfidf = pd.DataFrame.sparse.from_spmatrix(tfidf_matrix, 
                                                  columns=vectorizer.get_feature_names_out())

    # Remove stopwords
    df_tfidf = df_tfidf.loc[:, ~df_tfidf.columns.isin(stopwords_list)]

    return df_tfidf


def get_n_components(df_tfidf, show_viz=True):
    """
    Determine the number of components required to retain at least 90% of the 
    variance explained in a TF-IDF matrix using Singular Value Decomposition 
    (SVD) and visualize the variance explained interactively.

    Parameters
    ----------
    df_tfidf : pandas.DataFrame
        TF-IDF matrix represented as a DataFrame.

    Returns
    -------
    int
        The number of components required to explain at least 90% of the 
        variance.
    """
    svd = TruncatedSVD(n_components=len(df_tfidf.columns))
    df_svd = svd.fit_transform(df_tfidf)

    # Find the number of components that retain at least 90% of the variance explained
    indiv_var = svd.explained_variance_ratio_
    cum_var_explained = np.cumsum(indiv_var)
    n_components_90 = np.argmax(cum_var_explained >= 0.9) + 1
    
    if show_viz:
        print(f"Number of components for at least 90% variance explained: {n_components_90}")
        # Plot the variance explained interactively
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(1, len(indiv_var) + 1), y=indiv_var, mode='lines+markers', name='Individual', line=dict(color='black')))
        fig.add_trace(go.Scatter(x=np.arange(1, len(indiv_var) + 1), y=cum_var_explained, mode='lines+markers', name='Cumulative'))

        # Add a vertical line at 90% variance explained
        fig.add_shape(type='line', x0=n_components_90, y0=0, x1=n_components_90, y1=1, 
                      line=dict(color='black', width=1, dash='dash'))

        # Update layout to remove background color and increase plot size
        fig.update_layout(title=dict(text='Variance Explained by Singular Values', font=dict(color='black')),
                          xaxis=dict(title='Number of Components (Singular Values)', title_font=dict(color='black')),
                          yaxis=dict(title='Variance Explained', title_font=dict(color='black')),
                          legend=dict(x=0.05, y=0.95),
                          hovermode='x',
                          plot_bgcolor='rgba(0,0,0,0)',
                          paper_bgcolor='rgba(0,0,0,0)',
                          height=600,  # Adjust the height as needed
                          width=800)   # Adjust the width as needed

        fig.show()
    return n_components_90


def reduce_dimensions(n_components, df_tfidf):
    """
    Reduce the dimensions of a TF-IDF matrix using Singular Value 
    Decomposition (SVD).

    Parameters
    ----------
    n_components : int
        The number of components to reduce the matrix to.
    df_tfidf : pandas.DataFrame
        TF-IDF matrix represented as a DataFrame.

    Returns
    -------
    numpy.ndarray
        The TF-IDF matrix with reduced dimensions.
    """
    svd = TruncatedSVD(n_components=n_components)
    df = svd.fit_transform(df_tfidf)
    return df



def hierarchical_clustering(reduced_df, return_Z=False, show_viz=True):
    """
    Perform hierarchical clustering on the reduced dataset and identify the 
    optimal number of clusters.

    Parameters
    ----------
    reduced_df : pandas.DataFrame
        The reduced dataset after dimensionality reduction.
    return_Z : bool, optional
        If True, return the linkage matrix (Z) along with the threshold 
        distance. Default is False.
    show_viz : bool, optional
        If True, display the dendrogram visualization. Default is True.

    Returns
    -------
    int or tuple
        If return_Z is False (default), returns the threshold distance used to 
        determine clusters.
        If return_Z is True, returns a tuple containing the linkage matrix (Z)
        and the threshold distance.
    """
    # Perform hierarchical clustering using Ward's method
    linkage_matrix = linkage(reduced_df, method='ward', optimal_ordering=True)
    
    # Determine the optimal number of clusters and threshold distance
    distances = linkage_matrix[:, 2]
    gaps = np.diff(distances)
    highest_gap_index = np.argmax(gaps)
    num_clusters = len(distances) - highest_gap_index
    threshold = np.mean((distances[highest_gap_index:highest_gap_index + 2]))
    
    if show_viz:
        # Visualize the dendrogram
        fig, ax = plt.subplots(dpi=250)
        dendrogram(linkage_matrix, ax=ax, truncate_mode='level', p=5)
        ax.set_ylabel(r"Distance ($\Delta$)")
        # ax.hline()
       
        # Print the results
        print(f'Optimal number of clusters: {num_clusters}')
        print(f'Threshold distance: {threshold:.6f}')
    
    if return_Z:
        return linkage_matrix, threshold
    else:
        return threshold
    
def cluster_predict(reduced_df, main_df, show_viz=True, get_updated_df=False,
                    thres=None):
    """
    Predict clusters for the reduced dataset and update the main 
    DataFrame accordingly.

    Parameters
    ----------
    reduced_df : pandas.DataFrame
        The reduced dataset after dimensionality reduction.
    main_df : pandas.DataFrame
        The main DataFrame where cluster labels will be assigned.
    show_viz : bool, optional
        If True, display a scatter plot visualization of the predicted 
        clusters. Default is True.
    get_updated_df : bool, optional
        If True, return the main DataFrame with the cluster labels assigned. 
        Default is False.
    thres : float or None, optional
        The threshold distance used for clustering. If None, it will be 
        computed automatically based on the largest gap. Default is None.


    Returns
    -------
    pandas.DataFrame or None
        If get_updated_df is True, returns the main DataFrame with the cluster 
        labels assigned.
        Otherwise, returns None.
    """
    if thres is None:
        Z, thres = hierarchical_clustering(reduced_df, return_Z=True, 
                                           show_viz=False)
    else:
        Z = hierarchical_clustering(reduced_df, return_Z=True, 
                                    show_viz=False)[0]
        
    y_predict = fcluster(Z, t=thres, criterion="distance")
   
    if show_viz:
        custom_palette = ['#FF5733', '#FF8C00', '#FFC300', '#FFD700', 
                          '#FFD700', '#FFB6C1', '#FF69B4', '#CD5C5C', 
                          '#9370DB', '#00BFFF', '#7FFFD4', '#32CD32', 
                          '#6B8E23', '#FFD700']

        plt.figure(dpi=250)
        plt.scatter(reduced_df[:, 0], reduced_df[:, 1], alpha=0.5, c=y_predict,
                   cmap=plt.cm.colors.ListedColormap(custom_palette))
        plt.xlabel('SV2')
        plt.ylabel('SV1')
        plt.show()
    
    main_df['cluster'] = y_predict
    if get_updated_df:
        return main_df
    
def generate_wordclouds(clustered_df, cluster_number=1):
    """
    Generate and visualize WordClouds for abstracts and titles in two rows.

    Parameters
    ----------
    clustered_df : pandas.DataFrame
        DataFrame containing clustered data including abstracts and titles.
    cluster_number : int, optional
        The cluster number for which to generate the WordClouds.
        Default is 1.

    Returns
    -------
    None
        Displays the WordCloud visualization.
    """
    # Create subplots for each WordCloud
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), dpi=250)

    # Word cloud of abstracts
    abstract_words = (clustered_df[clustered_df["cluster"]
                                  == cluster_number]["abstract"]
                      .values
                      .tolist())
    filtered_abstract_words = []
    for word in abstract_words:
        word_list = word.split()
        filtered_word_list = [
            w for w in word_list if w.lower() not in stopwords_list]
        filtered_word = " ".join(filtered_word_list)
        filtered_abstract_words.append(filtered_word)
    wordcloud_abstracts = (WordCloud(width=800, height=400,
                                     background_color="white", color_func=lambda *args, **kwargs: '#7d0a0a')
                           .generate(" ".join(filtered_abstract_words)))
    axes[0].imshow(wordcloud_abstracts, interpolation="bilinear")
    axes[0].set_title('Word Cloud of Abstracts')
    axes[0].axis("off")

    # Add some space between the subplots
    plt.subplots_adjust(hspace=0.15)

    # Word cloud of titles
    title_words = clustered_df[clustered_df["cluster"]
                               == cluster_number]["title"].values.tolist()
    filtered_title_words = []
    for word in title_words:
        word_list = word.split()
        filtered_word_list = [
            w for w in word_list if w.lower() not in stopwords_list]
        filtered_word = " ".join(filtered_word_list)
        filtered_title_words.append(filtered_word)
    wordcloud_titles = (WordCloud(width=800, height=400,
                                  background_color="white", color_func=lambda *args, **kwargs: '#7d7463')
                        .generate(" ".join(filtered_title_words)))
    axes[1].imshow(wordcloud_titles, interpolation="bilinear")
    axes[1].set_title('Word Cloud of Titles')
    axes[1].axis("off")

    plt.show()
    
def get_sample(main_df, years, sample_size=2000):
    """
    Get a sample of data from the main DataFrame for each specified year.

    Parameters
    ----------
    main_df : pandas.DataFrame
        The main DataFrame containing the data.
    years : list of int
        List of years from which to sample data.
    sample_size : int, optional
        The number of rows to sample for each year. Default is 2000.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the sampled data from the specified years.
    """
    sampled_data_list = []
    for year in years:
        sampled_data = main_df[main_df.year == year].sample(n=sample_size, random_state=42)
        sampled_data_list.append(sampled_data)
        
    new_df = pd.concat(sampled_data_list)
    return new_df