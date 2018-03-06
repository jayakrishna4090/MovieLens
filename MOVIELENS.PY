# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:17:43 2018

@author: Jayakrishna
"""

import pandas as pd
import numpy as np

movie=pd.read_csv("D:\\ML\\MovieLens_Datasets\\ml-movies.csv")
rating=pd.read_csv("D:\\ML\\MovieLens_Datasets\\ml-ratings.csv")

year.clear()
year=[]
import re
for each in movie['title']:
    #print(re.search("\([0-9][0-9][0-9][0-9]\)",each))
    #value= re.search(r'\[([0-9][0-9][0-9][0-9])\]', each)
    #if value:
        
    if not re.findall(r"\([0-9]{4}\)", each):
        year.append(np.NaN)
    else:
        year.append(int(re.findall(r"\([0-9]{4}\)", each)[0][1:5]))
        #print(movie.loc[movie['title']==each])
    #print(movie.loc[re.findall(r"[0-9]{4}", each)[0]==''])
    #year.append(int(re.findall(r"[0-9]{4}", each)[0]))
movie['year']=year
    #year.append(int(re.search(r'\d+', each)))
"""count=[]
for each in movie['genres']:
    count.append(each.count("|"))
    if each.count("|")==0:
        print(each)
    
movie.loc[movie['genres'].count("|")==9]

for each in movie['genres']:
    if(re.finditer(r"|",each)[0]==0):
        print(each)"""

for each in movie['genres']:
#movie[['genre1','genre2','genre3','genre4','genre5','genre6','genre7','genre8','genre9','genre10']]=movie['genres'].str.split("|",maxsplit=1)
#value=sample.split("|\s+",maxsplit=True)
    if each.count("|")==0:
        movie.loc[movie['genres']==each,'genre1']=each
    else:
        for var in range(0,(each.count("|")+1)):
            movie.loc[movie['genres']==each,'genre'+str(var+1)]=each.split("|")[var]
            
  """          
    elif each.count("|")==1:
        movie.loc[movie['genres']==each,'genre1']=each.split("|")[0]
        movie.loc[movie['genres']==each,'genre2']=each.split("|")[1]      
    elif each.count("|")==2:
        movie.loc[movie['genres']==each,'genre1']=each.split("|")[0]
        movie.loc[movie['genres']==each,'genre2']=each.split("|")[1]
        movie.loc[movie['genres']==each,'genre3']=each.split("|")[2]
    elif each.count("|")==3:
        movie.loc[movie['genres']==each,'genre1']=each.split("|")[0]
        movie.loc[movie['genres']==each,'genre2']=each.split("|")[1]
        movie.loc[movie['genres']==each,'genre3']=each.split("|")[2]
        movie.loc[movie['genres']==each,'genre4']=each.split("|")[3]
    elif each.count("|")==4:
        movie.loc[movie['genres']==each,'genre1']=each.split("|")[0]
        movie.loc[movie['genres']==each,'genre2']=each.split("|")[1]
        movie.loc[movie['genres']==each,'genre3']=each.split("|")[2]
        movie.loc[movie['genres']==each,'genre4']=each.split("|")[3]
        movie.loc[movie['genres']==each,'genre5']=each.split("|")[4]
    elif each.count("|")==5:
        movie.loc[movie['genres']==each,'genre1']=each.split("|")[0]
        movie.loc[movie['genres']==each,'genre2']=each.split("|")[1]
        movie.loc[movie['genres']==each,'genre3']=each.split("|")[2]
        movie.loc[movie['genres']==each,'genre4']=each.split("|")[3]
        movie.loc[movie['genres']==each,'genre5']=each.split("|")[4]
        movie.loc[movie['genres']==each,'genre6']=each.split("|")[5]
    elif each.count("|")==6:
        movie.loc[movie['genres']==each,'genre1']=each.split("|")[0]
        movie.loc[movie['genres']==each,'genre2']=each.split("|")[1]
        movie.loc[movie['genres']==each,'genre3']=each.split("|")[2]
        movie.loc[movie['genres']==each,'genre4']=each.split("|")[3]
        movie.loc[movie['genres']==each,'genre5']=each.split("|")[4]
        movie.loc[movie['genres']==each,'genre6']=each.split("|")[5]
        movie.loc[movie['genres']==each,'genre7']=each.split("|")[6]
    elif each.count("|")==7:
        movie.loc[movie['genres']==each,'genre1']=each.split("|")[0]
        movie.loc[movie['genres']==each,'genre2']=each.split("|")[1]
        movie.loc[movie['genres']==each,'genre3']=each.split("|")[2]
        movie.loc[movie['genres']==each,'genre4']=each.split("|")[3]
        movie.loc[movie['genres']==each,'genre5']=each.split("|")[4]
        movie.loc[movie['genres']==each,'genre6']=each.split("|")[5]
        movie.loc[movie['genres']==each,'genre7']=each.split("|")[6]
        movie.loc[movie['genres']==each,'genre8']=each.split("|")[7]
    elif each.count("|")==8:
        movie.loc[movie['genres']==each,'genre1']=each.split("|")[0]
        movie.loc[movie['genres']==each,'genre2']=each.split("|")[1]
        movie.loc[movie['genres']==each,'genre3']=each.split("|")[2]
        movie.loc[movie['genres']==each,'genre4']=each.split("|")[3]
        movie.loc[movie['genres']==each,'genre5']=each.split("|")[4]
        movie.loc[movie['genres']==each,'genre6']=each.split("|")[5]
        movie.loc[movie['genres']==each,'genre7']=each.split("|")[6]
        movie.loc[movie['genres']==each,'genre8']=each.split("|")[7]
        movie.loc[movie['genres']==each,'genre9']=each.split("|")[8]
    elif each.count("|")==9:
        movie.loc[movie['genres']==each,'genre1']=each.split("|")[0]
        movie.loc[movie['genres']==each,'genre2']=each.split("|")[1]
        movie.loc[movie['genres']==each,'genre3']=each.split("|")[2]
        movie.loc[movie['genres']==each,'genre4']=each.split("|")[3]
        movie.loc[movie['genres']==each,'genre5']=each.split("|")[4]
        movie.loc[movie['genres']==each,'genre6']=each.split("|")[5]
        movie.loc[movie['genres']==each,'genre7']=each.split("|")[6]
        movie.loc[movie['genres']==each,'genre8']=each.split("|")[7]
        movie.loc[movie['genres']==each,'genre9']=each.split("|")[8]
        movie.loc[movie['genres']==each,'genre10']=each.split("|")[9]
            
    
    movie['genre1']=each.split("|")[0]
    movie['genre2']=each.split("|")[1]
    movie['genre3']=each.split("|")[2]
    movie['genre4']=each.split("|")[3]
    movie['genre5']=each.split("|")[4]
    movie['genre6']=each.split("|")[5]
    movie['genre7']=each.split("|")[6]
    movie['genre8']=each.split("|")[7]
    movie['genre9']=each.split("|")[8]
    movie['genre10']=each.split("|")[9]
sample="Action"


print(re.finditer(r"|","Action|Adventure|Comedy|Crime|Drama|Film-Noir|Horror|Mystery|Thriller|Western"))

text = "" 
sample.count("|")
firstTest = text.find('|')

print (text.find('|', firstTest ))

sample.split()"""

genlist.clear()
genlist=[]
for each in movie.columns:
    if re.findall(r"genre[0-9]",each):
        genlist.append(each)
total_genre=list(pd.unique(movie[genlist].values.ravel('K')))
total_genre.remove(np.nan)
#total_genre[19]='Undefined'
total_genre.sort
len(total_genre)

for each in total_genre:
    print(each)
    """
for value in movie['genres']:

            if([var].value==each):
                movie.loc[each]=1
    if movie['genre1']=='Adventure':
        print(movie.loc[value])
    for each in movie['genre1']:
        print(each)
    print(each)
    for each in total_genre:
        for col in genlist:
            if each==col:
                """"
for each in total_genre:
    for var in genlist:
        #movie.loc[movie[var]!=each,each]=0
        movie.loc[movie[var]==each,each]=1
            
movie=movie.rename(columns={'(no genres listed)':'Undefined'})

movie.drop(genlist,axis=1,inplace=True)

movie.loc[movie['Undefined']==1].shape

#22 year values has null - need to find the alternative

data=pd.merge(rating,movie,how='inner', on ='movieId').sort_values('userId')

#Average Rating
data['rating'].mean()

user_rating=data.pivot_table(data, 'rating', columns=['Comedy'])

5.dot(4.T)

import pyspark
