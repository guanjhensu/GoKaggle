# Read CSV file into DataFrame 
songs = read.csv('songs.csv')

#take a quick look into "songs.csv"
head(songs)


# Songs of 2010
sum(songs$year==2010)
table(songs$year)
# Songs from "Michael Jackson"
sum(songs$artistname == 'Michael Jackson')
MJ = subset(songs,artistname == 'Michael Jackson')
# Michael Jackson & Top10
subset(MJ,Top10==1)
# timesignature values
unique(songs$timesignature)
# most frequent timesignature 
table(songs$timesignature)
# song with the highest tempo
songs$songtitle[which.max(songs$tempo)]






#-----------------------#
#create prediction model#
#-----------------------#

# prepare train data & test data
nonvars = c("year","songtitle","artistname","songID","artistID")
# training data (1990~2009)
SongsTrain = subset(songs,year<=2009)

library(dplyr)
# Top10:nonTop10 = 1000:1000
SongTrain1 = subset(SongsTrain,Top10==0)
SongTrain2 = subset(SongsTrain,Top10==1)
s1 = sample_n(SongTrain1,1000)
s2 = sample_n(SongTrain2,1000)
newTrain = rbind(s1,s2)


# testing data (2010)
SongsTest = subset(songs,year==2010)
# eliminate features we don't use
newTrain = newTrain[,!(names(newTrain) %in% nonvars)]
SongsTrain = SongsTrain[,!(names(SongsTrain) %in% nonvars)]
SongsTest = SongsTest[,!(names(SongsTest) %in% nonvars)]


# Create Model
# Method one: use SongsTrain
SongsLog1 = glm(Top10 ~ ., data=SongsTrain, family=binomial)
SongsLog2 = glm(Top10 ~ .-loudness,data=SongsTrain, family=binomial)
SongsLog3 = glm(Top10 ~ .-energy, data=SongsTrain, family=binomial)

# Method two: use newTrain
SongsLog1 = glm(Top10 ~ ., data=newTrain, family=binomial)
SongsLog2 = glm(Top10 ~ .-loudness,data=newTrain, family=binomial)
SongsLog3 = glm(Top10 ~ .-energy, data=newTrain, family=binomial)

# Evaluation : confusion matrix
testPredict = predict(SongsLog3, newdata=SongsTest, type="response")
confusion = table(SongsTest$Top10, testPredict >= 0.5) #row為實際，column為預測方向(右邊參數為判斷T or F)
sensitivity = confusion[4]/(confusion[2]+confusion[4])
specificity = confusion[1]/(confusion[1]+confusion[3])





SongsLog4 = glm(formula = Top10 ~ timesignature_confidence + loudness + key_confidence + 
      pitch + timbre_0_min + timbre_0_max + timbre_1_min + timbre_2_min + 
      timbre_3_max + timbre_4_min + timbre_4_max + timbre_5_min + 
      timbre_6_min + timbre_6_max + timbre_7_min + timbre_10_min + 
      timbre_10_max + timbre_11_min + timbre_11_max, family = binomial, 
    data = newTrain)
summary(SongsLog4) 










#gradient descent
scale = function(x){
  scalex = (x-mean(x))/(max(x)-min(x))
  return(scalex)
}


nonvars = c("year","songtitle","artistname","songID","artistID","loudness")
SongsTrain = SongsTrain[,!(names(SongsTrain) %in% nonvars)]

x = as.matrix(SongsTrain[,-33])
nscalex = cbind(x0 = 1,x)
newx = apply(x,2,scale)
scalex = cbind(x0 = 1,newx)

y = SongsTrain$Top10
m = length(SongsTrain$timesignature)
theta = as.matrix(rep(1,times=33))
alpha = 0.8

gradient = function(theta,alpha,x,y,m){
h = 1/(1+exp(-x %*% theta))
e = h-y
i = 1
jfunc = c()
cost = -(1/m)*(t(y) %*% log(h) + t(1-y) %*% log(1-h))
dcost = (1/m)*(t(x) %*% e)
jfunc[i] = cost
theta = theta - alpha * dcost
i = i + 1
while(TRUE){
  h = 1/(1+exp(-x %*% theta))
  e = h - y
  cost = -(1/m)*(t(y) %*% log(h) + t(1-y) %*% log(1-h))
  dcost = (1/m)*(t(x) %*% e) 
  jfunc[i] = cost
  if(abs(jfunc[i] - jfunc[i-1])<=1e-15) break
  theta = theta - alpha * dcost
  i = i + 1
}
return(list(theta=theta,cost=cost))
}
theta1 = as.matrix(SongsLog2$coefficients)
h1 = 1/(1+exp(-nscalex %*% theta1))
cost = -(1/m)*(t(y) %*% log(h1) + t(1-y) %*% log(1-h1)) 
#answer:0.3382759

gradient(theta,alpha = 0.1,scalex,y,m)




h = 1/(1+exp(-scalex %*% theta))
e = h-y
i = 1
jfunc = c()
cost = -(1/m)*(t(y) %*% log(h) + t(1-y) %*% log(1-h))
dcost = (1/m)*(t(scalex) %*% e)
jfunc[i] = cost
alpha = 1
theta = theta - alpha * dcost
i = i + 1
while(TRUE){
  h = 1/(1+exp(-scalex %*% theta))
  e = h - y
  cost = -(1/m)*(t(y) %*% log(h) + t(1-y) %*% log(1-h))
  dcost = (1/m)*(t(scalex) %*% e) 
  jfunc[i] = cost
  print(cost)
  if(abs(jfunc[i] - jfunc[i-1])<=1e-15) break
  theta = theta - alpha * dcost
  i = i + 1
}









