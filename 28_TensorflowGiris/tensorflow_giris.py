import keras.models
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error



dataFrame = pd.read_excel("bisiklet_fiyatlari.xlsx")
print(dataFrame)
print(dataFrame.head()) #ilk 5 veriyi getiriyor

sbn.pairplot(dataFrame) #farklı grafiklerde gösterim
plt.show()
#veriyi test/train olarak ikiye ayırmak

#train_test_split
#y = wx + bd

#y -> label
y = dataFrame["Fiyat"].values

#x -> feature(özellik)
x = dataFrame[["BisikletOzellik1","BisikletOzellik2"]].values

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.33, random_state = 15)
print(x_train)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
#buraya kadar veriyi alıp ikiye böldük

#scaling boyut ayarlama
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
#verileri 0 ile 1 arasına getirildi
#model oluşturma;
model = Sequential()
model.add(Dense(4,activation="relu"))
model.add(Dense(4,activation="relu"))
model.add(Dense(4,activation="relu"))

model.add(Dense(1))
model.compile(optimizer="rmsprop",loss="mse")
#training;
model.fit(x_train,y_train,epochs=250,batch_size=1)
print(model.history.history)
loss = model.history.history["loss"]
sbn.lineplot(x=range(len(loss)),y=loss)
plt.show()

trainLoss = model.evaluate(x_train,y_train,verbose=0)
testLoss = model.evaluate(x_test,y_test,verbose=0)
print(trainLoss)
print(testLoss) #birbirlerine yakın olmaları önemli

testTahminleri = model.predict(x_test)
print(testTahminleri) #dizi

tahminDF = pd.DataFrame(y_test,columns=["Gerçek Y"])
print(tahminDF)

testTahminleri = pd.Series(testTahminleri.reshape(330,))
print(testTahminleri)

tahminDF = pd.concat([tahminDF,testTahminleri],axis=1)
tahminDF.columns = ["Gerçek Y", "Tahmin Y"]
print(tahminDF)

sbn.scatterplot(x = "Gerçek Y", y = "Tahmin Y", data=tahminDF)
plt.show()

print(mean_absolute_error(tahminDF["Gerçek Y"],tahminDF["Tahmin Y"])) #hata hesaplama
print(mean_squared_error(tahminDF["Gerçek Y"],tahminDF["Tahmin Y"]))
print(dataFrame.describe())

yeniBisikletOzellikleri = [[1751,1750]]
yeniBisikletOzellikleri = scaler.transform(yeniBisikletOzellikleri)
print(model.predict(yeniBisikletOzellikleri))

model.save("bisiklet_modeli.h5") #kaydetme
sonradanCagirilanModel = load_model("bisiklet_modeli.h5") #değişkene model yükleme
print(sonradanCagirilanModel.predict(yeniBisikletOzellikleri))


