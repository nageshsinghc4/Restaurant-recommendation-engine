#collaborative filtering for blog
import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


review_df = pd.read_csv('/Users/nageshsinghchauhan/Downloads/ML/recommend/yelp_review.csv', encoding="latin-1")
review_df = review_df[['user_id', 'business_id', 'stars']]
review_df = review_df.rename(columns = {'stars':'usr_rating'})
review_df.dropna(inplace=True)

business_df = pd.read_csv('/Users/nageshsinghchauhan/Downloads/ML/recommend/yelp_business.csv', encoding="latin-1")
business_df = business_df[['business_id', 'name', 'city', 'stars', 'review_count', 'categories']]
business_df = business_df.rename(columns = {'stars':'restaurant_rating'})

#include category which shows restaurants
business_df = business_df[business_df['categories'].str.contains("Food|Coffee|Tea|Restaurants|Bakeries|Bars|Sports Bar|Pubs|Nighlife")]
business_df.dropna(inplace=True)

#data visualization
#Get the distribution of the restaurant rating
plt.figure(figsize=(12,4))
ax = sns.countplot(business_df['restaurant_rating'])
plt.title('Distribution of Restaurant Rating');

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()
import plotly.graph_objs as go
#Top 10 most reviewed businesses¶
business_df[['name', 'review_count', 'city', 'restaurant_rating']].sort_values(ascending=False, by="review_count")[0:10]
business_df['name'].value_counts().sort_values(ascending=False).head(10)
business_df['name'].value_counts().sort_values(ascending=False).head(10).plot(kind='pie',figsize=(10,6),
title="Most Popular Cuisines", autopct='%1.2f%%')
plt.axis('equal')

#how many different business registered on yelp
fig, ax = plt.subplots(figsize=[5,10])
sns.countplot(data=business_df[business_df['categories'].isin(
    business_df['categories'].value_counts().head(25).index)],
                              y='categories', ax=ax)
plt.show()

#Number of businesses listed in different cities
city_business_counts = business_df[['city', 'business_id']].groupby(['city'])\
['business_id'].agg('count').sort_values(ascending=False)
city_business_counts = pd.DataFrame(data=city_business_counts)
city_business_counts.rename(columns={'business_id' : 'number_of_businesses'}, inplace=True)
city_business_counts[0:10].sort_values(ascending=True, by="number_of_businesses")\
.plot(kind='bar', stacked=False, figsize=[10,10], colormap='winter')
plt.title('Top 50 cities by businesses listed')

#Cities with most reviews and best ratings for their businesses
city_business_reviews = business_df[['city', 'review_count', 'restaurant_rating']].groupby(['city']).\
agg({'review_count': 'sum', 'restaurant_rating': 'mean'}).sort_values(by='review_count', ascending=False)
city_business_reviews.head(10)
city_business_reviews['review_count'][0:10].plot(kind='bar', stacked=False, figsize=[10,10], \
                                                  colormap='winter')
plt.title('Top 50 cities by reviews')

#end of data exploration and visualization

joined_restaurant_rating = pd.merge(business_df, review_df, on='business_id')
restaurant_ratingCount = (joined_restaurant_rating.
     groupby(by = ['name'])['restaurant_rating'].
     count().
     reset_index().
     rename(columns = {'restaurant_rating': 'totalRatingCount'})
     [['name', 'totalRatingCount']]
    )

rating_with_totalRatingCount = joined_restaurant_rating.merge(restaurant_ratingCount, left_on = 'name', right_on = 'name', how = 'left')

pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(rating_with_totalRatingCount['totalRatingCount'].describe())

# Calculate the minimum number of votes required to be in the chart, m
populatity_threshold = rating_with_totalRatingCount['totalRatingCount'].quantile(0.90)

rating_popular_rest = rating_with_totalRatingCount.query('totalRatingCount >= @populatity_threshold')

#Select top
us_city_user_rating = rating_popular_rest[rating_popular_rest['city'].str.contains("Las Vegas|Pheonix|Toronto|Scattsdale|Charlotte|Tempe|Chandler|Cleveland|Madison|Gilbert")]
#us_canada_user_rating.head()
us_city_user_rating = us_city_user_rating.drop_duplicates(['user_id', 'name'])
restaurant_features = us_city_user_rating.pivot(index = 'name', columns = 'user_id', values = 'restaurant_rating').fillna(0)

from scipy.sparse import csr_matrix
restaurant_features_matrix = csr_matrix(restaurant_features.values)

from sklearn.neighbors import NearestNeighbors
knn_recomm = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
knn_recomm.fit(restaurant_features_matrix)

randomChoice = np.random.choice(restaurant_features.shape[0])
distances, indices = knn_recomm.kneighbors(restaurant_features.iloc[randomChoice].values.reshape(1, -1), n_neighbors = 11)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for Restaurant {0} on priority basis:\n'.format(restaurant_features.index[randomChoice]))
    else:
        print('{0}: {1}'.format(i, restaurant_features.index[indices.flatten()[i]]))
