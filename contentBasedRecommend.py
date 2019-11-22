#content-based recommendation system

import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.tokenize import word_tokenize
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/nageshsinghchauhan/Downloads/ML/recommend/zomato.csv', encoding ='latin1')


data['City'].value_counts(dropna = False).head(5)

data_city =data.loc[data['City'] == 'New Delhi']

data_new_delphi=data_city[['Restaurant Name','Cuisines','Locality','Aggregate rating']]

data_new_delphi['Locality'].value_counts(dropna = False).head(5)

data_new_delphi.loc[data['Locality'] == 'Connaught Place']

data_sample=[]

def restaurant_recommend_func(location,title):

    # these variable has to global because of i want use some properties out of function for analysis
    global data_sample
    global cosine_sim
    global sim_scores
    global tfidf_matrix
    global corpus_index
    global feature
    global rest_indices
    global idx

    # When location comes from function ,our new data consist only location dataset
    data_sample = data_new_delphi.loc[data_new_delphi['Locality'] == location]

    # index will be reset for cosine similarty index because Cosine similarty index has to be same value with result of tf-idf vectorize
    data_sample.reset_index(level=0, inplace=True)

    #Feature Extraction
    data_sample['Split']='X'
    for i in range(0,data_sample.index[-1]):
        split_data=re.split(r'[,]', data_sample['Cuisines'][i])
        for k,l in enumerate(split_data):
            split_data[k]=(split_data[k].replace(" ", ""))
        split_data=' '.join(split_data[:])
        data_sample['Split'].iloc[i]=split_data

    ## --- TF - IDF Vectorizer---  ##
    #Extracting Stopword
    tfidf = TfidfVectorizer(stop_words='english')


    #Replace NaN for empty string
    data_sample['Split'] = data_sample['Split'].fillna('')

    # Applying TF-IDF Vectorizer
    tfidf_matrix = tfidf.fit_transform(data_sample['Split'])
    tfidf_matrix.shape

    # Using for see Cosine Similarty scores
    feature= tfidf.get_feature_names()


    ## ---Cosine Similarity--- ##
    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Column names are using for index
    corpus_index=[n for n in data_sample['Split']]

    #Construct a reverse map of indices
    indices = pd.Series(data_sample.index, index=data_sample['Restaurant Name']).drop_duplicates()

    #index of the restaurant matchs the cuisines
    idx = indices[title]


    #Aggregate rating added with cosine score in sim_score list.
    sim_scores=[]
    for i,j in enumerate(cosine_sim[idx]):
        k=data_sample['Aggregate rating'].iloc[i]
        if j != 0 :
            sim_scores.append((i,j,k))

    #Sort the restaurant names based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: (x[1],x[2]) , reverse=True)

    # 5 similarty cuisines
    sim_scores = sim_scores[0:6]

    rest_indices = [i[0] for i in sim_scores]

    data_x =data_sample[['Restaurant Name','Cuisines','Aggregate rating']].iloc[rest_indices]

    data_x['Cosine Similarity']=0
    for i,j in enumerate(sim_scores):
        data_x['Cosine Similarity'].iloc[i]=round(sim_scores[i][1],2)

    return data_x

# Top 5 similar restaurant with cuisine of 'Barbeque Nation' restaurant in Connaught Place
restaurant_recommend_func('Connaught Place','Barbeque Nation')  ## location & Restaurant Name