# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 16:21:49 2018

@author: jthatha
"""


import pandas as pd
import numpy as np

movie=pd.read_csv("D:\\ML\\MovieLens_Datasets\\ml-movies.csv")
rating=pd.read_csv("D:\\ML\\MovieLens_Datasets\\ml-ratings.csv")


year=[]
import re
for each in movie['title']:
    if not re.findall(r"\([0-9]{4}\)", each):
        year.append(np.NaN)
    else:
        year.append(int(re.findall(r"\([0-9]{4}\)", each)[0][1:5]))
movie['year']=year


for each in movie['genres']:
    if each.count("|")==0:
        movie.loc[movie['genres']==each,'genre1']=each
    else:
        for var in range(0,(each.count("|")+1)):
            movie.loc[movie['genres']==each,'genre'+str(var+1)]=each.split("|")[var]

genlist=[]
for each in movie.columns:
    if re.findall(r"genre[0-9]",each):
        genlist.append(each)
total_genre=list(pd.unique(movie[genlist].values.ravel('K')))
total_genre.remove(np.nan)
total_genre.sort
len(total_genre)

for each in total_genre:
    for var in genlist:
        #movie.loc[movie[var]!=each,each]=0
        movie.loc[movie[var]==each,each]=1
            
movie=movie.rename(columns={'(no genres listed)':'Undefined'})

movie.drop(genlist,axis=1,inplace=True)

R_df = rating.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
R_df.head()

R = R_df.as_matrix()
user_ratings_mean = np.mean(R, axis = 1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)


from scipy.sparse.linalg import svds
U, sigma, Vt = svds(R_demeaned, k = 50)

sigma = np.diag(sigma)

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = R_df.columns)

def recommend_movies(predictions_df, userID, movies_df, original_ratings_df, num_recommendations=5):
    
    # Get and sort the user's predictions
    user_row_number = userID - 1 # UserID starts at 0, not 1
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)
    
    # Get the user's data and merge in the movie information.
    user_data = original_ratings_df[original_ratings_df.userId == (userID)]
    user_full = (user_data.merge(movies_df, how = 'left', left_on = 'movieId', right_on = 'movieId').
                     sort_values(['rating'], ascending=False)
                 )

    print ('User {0} has already rated {1} movies.'.format(userID, user_full.shape[0]))
    print ('Recommending the highest {0} predicted ratings movies not already rated.'.format(num_recommendations))
    
    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies_df[~movies_df['movieId'].isin(user_full['movieId'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'movieId',
               right_on = 'movieId').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations, :-1]
                      )

    return user_full, recommendations

already_rated, predictions = recommend_movies(preds_df, 1, movie, rating, 10)
