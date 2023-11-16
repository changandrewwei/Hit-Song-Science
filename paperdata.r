library(readr)
install.packages("caret")
library(caret)
install.packages("glmnet")
library(glmnet)
install.packages("boot")
library(boot)
library(pROC)

#更改工作路徑
setwd("D:/")
getwd()

#讀取資料集
songs = read_csv("../songs.csv")

spotifytop100 = read_csv("../spotifytop100.csv")

#進行表格合併
G3=merge(spotifytop100,songs,by=c("song","artist"),all.x = TRUE)

#將資料表格中的亂碼替換掉
G3$added=gsub("\\?","/",G3$added)

#將資料表中的的目標預測變量變成以0和1進行二元變數表示(沒進榜為0,進榜為1)
G3$rank[is.na(G3$rank)] = 0#將表格合併後的缺失值(N/A)改成以0表示
G3$rank[G3$rank!=0] = 1#將已經進榜的歌曲以1表示入榜(替換掉原來的排名)

#執行羅吉斯回歸將所有變數全部放入建立模型
P = glm(rank~pop+live+dB+bpm+nrgy+dur+dnce+acous+spch,data = G3,family = "binomial")

#進行逐步回歸(將影響顯著的模型放入)
finalml = step(P,scope = list(upper=~.),direction = "both")

#觀察模型並觀察哪些變數會明顯影響進榜與否
summary(finalml)

#逐步回歸後的模型點狀圖
plot(finalml)

#混淆矩陣
# 使用 finalml 進行預測，將含有預測變數的newdata放入finalml進行預測
prob = predict(finalml, newdata = G3, type = "response")

# 將預測概率轉換為二元預測（將數值大於0.5的預測為1，否則為0）
predicted = ifelse(prob > 0.5, 1, 0)

# 將實際觀測值轉換為二元變數
actual = G3$rank

troc=roc(actual~prob,plot=TRUE,print.auc=TRUE)

as.numeric(troc$auc)

#將兩者轉換成因子以進行混淆矩陣的運算
actual = factor(actual)

predicted = factor(predicted)

# 求出混淆矩陣後印出
CM = confusionMatrix(actual, predicted)

print(CM)




