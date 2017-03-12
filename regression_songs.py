# pandas : Python Data Analysis Library
import pandas as pd
from pandas import Series,DataFrame
import numpy as np

# Read CSV file into DataFrame
songs_df = pd.read_csv("songs.csv")

#take a quick look into "songs.csv" 
print(songs_df.head())
print(songs_df.tail(5))

# Songs of 2010
ans_1_1=songs_df[songs_df.year==2010].count()["year"] 
print(ans_1_1)
# Songs from "Michael Jackson"
ans_1_2=songs_df[songs_df.artistname=="Michael Jackson"].count()["artistname"]
print(ans_1_2)
# Michael Jackson & Top10
ans_1_3=songs_df.loc[(songs_df.artistname=="Michael Jackson")&(songs_df.Top10==1)]["songtitle"]
print(ans_1_3)
# timesignature values
ans_1_4_1=Series(songs_df.timesignature.values.ravel()).unique()
print(ans_1_4_1)
# most frequent timesignature 
ans_1_4_2=songs_df.groupby(songs_df.timesignature, sort=False).count()
print(ans_1_4_2)
# song with the highest tempo
max_tempo_idx=songs_df.tempo.idxmax(axis=0)
ans_1_5=songs_df.songtitle[max_tempo_idx]
print(ans_1_5)


#-----------------------#
#create prediction model#
#-----------------------#

# prepare train data & test data
nonvars=["year", "songtitle", "artistname", "songID", "artistID"]
'''
# Way 1: 
SongTrain=songs_df.drop(songs_df[songs_df.year==2010].index)
SongTrain=SongTrain.drop(nonvars, 1)
SongTest=songs_df.drop(songs_df[songs_df.year!=2010].index)
SongTest=SongTest.drop(nonvars, 1)
'''

# training data (1990~2009) 
# Top10:nonTop10 = 1000:1000
songs_df_train=songs_df.drop(songs_df[songs_df.year==2010].index).drop(nonvars, 1)
songs_df_Top10=songs_df_train.drop(songs_df_train[songs_df_train.Top10==0].index)
songs_df_not_Top10=songs_df_train.drop(songs_df_train[songs_df_train.Top10==1].index)
SongTrain=songs_df_Top10.sample(1000).append(songs_df_not_Top10.sample(1000))




# testing data (2010)
songs_df_test=songs_df.drop(songs_df[songs_df.year!=2010].index).drop(nonvars, 1)
SongTest=songs_df_test.sample(373)





# Create Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
logreg = LogisticRegression()
Y_ans=SongTest.Top10

# Model 1
X_train=SongTrain.drop("Top10",1)
Y_train=SongTrain.Top10
X_test1=SongTest.drop("Top10",1)
SongLog_1_score=logreg.fit(X_train, Y_train).score(X_test1, Y_ans)
print(SongLog_1_score)
Y_pred_1=logreg.predict(X_test1)
SongLog_1=logreg.fit(X_train, Y_train)
coeff_df_1 = DataFrame(X_train.columns)
coeff_df_1.columns = ['Features']
coeff_df_1["Coefficient Estimate"] = pd.Series(SongLog_1.coef_[0])
#print(coeff_df_1)
print(confusion_matrix(Y_ans, Y_pred_1))

# Model 2
SongLog_2_score=logreg.fit(X_train.drop("loudness", 1), Y_train).score(X_test1.drop("loudness",1), Y_ans)
print(SongLog_2_score)
Y_pred_2=logreg.predict(X_test1.drop("loudness",1))
SongLog_2=logreg.fit(X_train.drop("loudness", 1), Y_train)
coeff_df_2 = DataFrame(X_train.drop("loudness", 1).columns)
coeff_df_2.columns = ['Features']
coeff_df_2["Coefficient Estimate"] = pd.Series(SongLog_2.coef_[0])
#print(coeff_df_2)
print(confusion_matrix(Y_ans, Y_pred_2))

# Model 3
SongLog_3_score=logreg.fit(X_train.drop("energy", 1), Y_train).score(X_test1.drop("energy",1), Y_ans)
print(SongLog_3_score)
Y_pred_3=logreg.predict(X_test1.drop("energy",1))
SongLog_3=logreg.fit(X_train.drop("energy", 1), Y_train)
coeff_df_3 = DataFrame(X_train.drop("energy", 1).columns)
coeff_df_3.columns = ['Features']
coeff_df_3["Coefficient Estimate"] = pd.Series(SongLog_3.coef_[0])
#print(coeff_df_3)
print(confusion_matrix(Y_ans, Y_pred_3))


