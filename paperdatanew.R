library(readr)
install.packages("caret")
library(caret)
install.packages("glmnet")
library(glmnet)
install.packages("boot")
library(boot)
library(pROC)
install.packages("randomForest")
library(randomForest)

#更改工作路徑

setwd("D:/")

getwd()

#讀取資料集

songs = read_csv("../songs.csv")

spotifytop100 = read_csv("../spotifytop100.csv")

youtube = read_csv("../Spotify_Youtube.csv")

#進行表格合併(all.x=TRUE保留左側變數)

G3=merge(spotifytop100,songs,by=c("song","artist"),all.x = TRUE)

G4=merge(G3,youtube,by=c("song","artist"),all.x = TRUE)

#將資料表格中的亂碼替換掉

G4$added=gsub("\\?","/",G4$added)

#將資料表中的的目標預測變量變成以0和1進行二元變數表示(沒進榜為0,進榜為1)

G4$rank[is.na(G4$rank)] = 0#將表格合併後的缺失值(N/A)改成以0表示

G4$rank[G4$rank!=0] = 1#將已經進榜的歌曲以1表示入榜(替換掉原來的排名)

#將Licensed跟official_video變成二元變數

G4$Licensed[is.na(G4$Licensed)] = 0

G4$Licensed[G4$Licensed==TRUE] = 1

G4$official_video[is.na(G4$official_video)] = 0

G4$official_video[G4$official_video==TRUE] = 1

#將字串型態轉換成數值型態

G4$Likes=as.numeric(G4$Likes)

G4$Comments=as.numeric(G4$Comments)

G4$Stream=as.numeric(G4$Stream)

G4$Licensed=as.numeric(G4$Licensed)

G4$official_video=as.numeric(G4$official_video)

#法1:資料的缺失值補0

G4$Likes[is.na(G4$Likes)] = 0

G4$Comments[is.na(G4$Comments)] = 0

G4$Stream[is.na(G4$Stream)] = 0

G4$Views[is.na(G4$Views)] = 0

#法2:將所有缺失值補0

G4[is.na(G4)] = 0

#法3:用平均值(Mean)來填補缺失值

#View欄位

G4M1 = mean(G4[, 31], na.rm = T)

na.rows1 = is.na(G4[, 31])

G4[na.rows1, 31] = G4M1

#Likes欄位

G4M2 = mean(G4[, 32], na.rm = T)

na.rows2 = is.na(G4[, 32])

G4[na.rows2, 32] = G4M2

#Comments欄位

G4M3 = mean(G4[, 33], na.rm = T)

na.rows3 = is.na(G4[, 33])

G4[na.rows3, 33] = G4M3

#Stream欄位

G4M4 = mean(G4[, 37], na.rm = T)

na.rows4 = is.na(G4[, 37])

G4[na.rows4, 37] = G4M4

#將資料分割成測試集和訓練集

set.seed(123)  # 設置隨機種子

split1 = sample.split(G4$rank, SplitRatio=0.7)

Trainset = G4[split1,]

Testset = G4[!split1,]

#執行羅吉斯回歸將所有變數全部放入建立模型

P = glm(rank~pop+live+dB+bpm+nrgy+dur+dnce+acous+spch+Views+Likes+Comments+Stream
        +Licensed+official_video,data = Trainset,family = "binomial")

#進行逐步回歸(將影響顯著的模型放入)

finalml = step(P,scope = list(upper=~.),direction = "both")

summary(finalml)

#使用caret中的train函數進行交叉驗證(羅吉斯回歸)

TC = trainControl(method = "repeatedcv",number = 10, repeats = 3)

model1 = train(rank~pop+live+dB+bpm+nrgy+dur+dnce+acous+spch+Views+Likes+Comments+Stream,
               data = Trainset, method = "glm",trControl = TC)

#輸出交叉驗證結果(羅吉斯回歸)

print(model1)

#用Testset(測試集)預測概率

pred =  predict(finalml, Testset, type="response")

#以0.5為界判斷測試集為1或0

predicted = pred > 0.5

# 將實際觀測值轉換為二元變數

actual = Testset$rank

#繪製混淆矩陣(羅吉斯回歸)

cm = table(actual, predicted)

print(cm)

#繪製ROC曲線(羅吉斯回歸)

troc=roc(actual~pred,plot=TRUE,print.auc=TRUE)

as.numeric(troc$auc)

#利用隨機森林預測

#將Trainset(訓練集)的缺失值補0

Trainset[is.na(Trainset)] = 0

#建立隨機森林模型

rm = randomForest(rank~pop+live+dB+bpm+nrgy+dur+dnce+acous+spch+Views+Likes+Comments+Stream
                  ,data = Trainset,ntree=1000)

#使用caret中的train函數進行交叉驗證(隨機森林)

model2 = train(rank~pop+live+dB+bpm+nrgy+dur+dnce+acous+spch+Views+Likes+Comments
               ,data = Trainset, method = "rf",trControl = TC,ntree = 1000)

#輸出交叉驗證結果(隨機森林)

print(model2)

#用Testset(測試集)預測概率

pred2 =  predict(rm, Testset, type="response")

#以0.5為界判斷測試集為1或0

predicted2 = pred2 > 0.5

#繪製混淆矩陣(隨機森林)

cm2 = table(actual, predicted2)

print(cm2)

#繪製ROC曲線(隨機森林)

troc2=roc(actual~pred2,plot=TRUE,print.auc=TRUE)

as.numeric(troc2$auc)
