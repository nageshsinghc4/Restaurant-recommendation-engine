#Simple Recommendation system
import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
#Download the dataset from
#https://www.kaggle.com/shrutimehta/zomato-restaurants-data
data = pd.read_csv('/Users/nageshsinghchauhan/Downloads/ML/recommend/zomato.csv', encoding ='latin1')
data.head()

country = pd.read_excel("/Users/nageshsinghchauhan/Downloads/ML/recommend/Country-Code.xlsx")

data1 = pd.merge(data, country, on='Country Code')
data1.head(3)
#Data exploration

#selecy india
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
import plotly.graph_objs as go

abels = list(data1.Country.value_counts().index)
values = list(data1.Country.value_counts().values)

fig = {
    "data":[
        {
            "labels" : labels,
            "values" : values,
            "hoverinfo" : 'label+percent',
            "domain": {"x": [0, .9]},
            "hole" : 0.6,
            "type" : "pie",
            "rotation":120,
        },
    ],
    "layout": {
        "title" : "Zomato's Presence around the World",
        "annotations": [
            {
                "font": {"size":20},
                "showarrow": True,
                "text": "Countries",
                "x":0.2,
                "y":0.9,
            },
        ]
    }
}

iplot(fig)


#Top 5 cities in India where maximum number of restaurants are registered on zomato¶
res_India = data1[data1.Country == 'India']

#
labels1 = list(res_India.City.value_counts().index)
values1 = list(res_India.City.value_counts().values)
labels1 = labels1[:10]
values1 = values1[:10]


fig = {
    "data":[
        {
            "labels" : labels1,
            "values" : values1,
            "hoverinfo" : 'label+percent',
            "domain": {"x": [0, .9]},
            "hole" : 0.6,
            "type" : "pie",
            "rotation":120,
        },
    ],
    "layout": {
        "title" : "Zomato's Presence in India",
        "annotations": [
            {
                "font": {"size":20},
                "showarrow": True,
                "text": "Cities",
                "x":0.2,
                "y":0.9,
            },
        ]
    }
}

iplot(fig)

top5 = res_India.City.value_counts().head()
f , ax = plt.subplots(1,1,figsize = (8,4))
ax = sns.barplot(top5.index,top5,palette ='Set1')
plt.show()

#Top 10 Cuisines served by restaurants
res_India['Cuisines'].value_counts().sort_values(ascending=False).head(10)
res_India['Cuisines'].value_counts().sort_values(ascending=False).head(10).plot(kind='pie',figsize=(10,6),
title="Most Popular Cuisines", autopct='%1.2f%%')
plt.axis('equal')
#Number of restaurants in NCR with aggregate rating ranging from 1.9 to 4.9
NCR = ['New Delhi','Gurgaon','Noida','Faridabad']
res_NCR = res_India[(res_India.City == NCR[0])|(res_India.City == NCR[1])|(res_India.City == NCR[2])|
                    (res_India.City == NCR[3])]
res_NCR.head(3)


#data['City'].value_counts(dropna = False)

#data_city =data.loc[data['City'] == 'New Delhi']

data_new_delphi=res_NCR[['Restaurant Name','Cuisines','Locality','Aggregate rating', 'Votes']]

C = data_new_delphi['Aggregate rating'].mean()
print(C)
# avg rating is 2.43884524027

# Calculate the minimum number of votes required to be in the chart, m
m = data_new_delphi['Votes'].quantile(0.90)
print(m)

# Filter out all qualified restaurants into a new DataFrame
q_restaurant = data_new_delphi.copy().loc[data_new_delphi['Votes'] >= m]
q_restaurant.shape


# Function that computes the weighted rating of each movie
def weighted_rating(x, m=m, C=C):
    v = x['Votes']
    R = x['Aggregate rating']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_restaurant['score'] = q_restaurant.apply(weighted_rating, axis=1)

#Sort movies based on score calculated above
q_restaurant = q_restaurant.sort_values('score', ascending=False)

#Print the top 15 res
q_restaurant[['Restaurant Name','Cuisines', 'Locality','Votes', 'Aggregate rating', 'score']].head(10)