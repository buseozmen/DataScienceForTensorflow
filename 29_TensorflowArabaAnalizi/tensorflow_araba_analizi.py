import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

dataFrame = pd.read_excel("merc.xlsx")
print(dataFrame.head()) #ilk 5 veri
#veriyi anlamak
print(dataFrame.describe())
print(dataFrame.isnull().sum()) #kaç boşluk var

#grafiksel analizler
plt.figure(figsize=(7,5))
sbn.histplot(dataFrame["price"]) #displot
plt.show()

sbn.countplot(x = dataFrame["year"])
plt.show()

dataFrame.drop("transmission",axis=1,inplace=True) #corr hata verdiği için sildik bu kısmı
print(dataFrame.corr())
print(dataFrame.corr()["price"].sort_values()) #değerleri sırala

sbn.scatterplot(x="mileage",y="price",data=dataFrame)
plt.show()

#en yüksek fiyatlı arabalar
print(dataFrame.sort_values("price",ascending=False).head(20)) #fiyatı azalan bir şekilde
print(dataFrame.sort_values("price",ascending=True).head(20)) #fiyatı yükselen bir şekilde

print(len(dataFrame))
print(len(dataFrame) * 0.01)

#veri temizliği
yuzdeDoksanDokuzDF = pd.DataFrame(dataFrame.sort_values("price",ascending=False).iloc[131:]) #131. index sonrası
print(yuzdeDoksanDokuzDF.describe())
plt.figure(figsize=(7,5))
sbn.histplot(yuzdeDoksanDokuzDF["price"]) #daha normal artan ve azalan bir grafik
plt.show()

print(dataFrame.groupby("year").mean()["price"]) #ortalama bir değer gösteriyor
print(yuzdeDoksanDokuzDF.groupby("year").mean()["price"]) #pek fazla bir fark yok orj dataset ile
#gereksiz olan 1970 de ki arabaları çıkarmış gibi gösterelim
print(dataFrame[dataFrame.year != 1970].groupby("year").mean()["price"]) #henüz çıkarılmadı sadece gösterdik

dataFrame = yuzdeDoksanDokuzDF
print(dataFrame.describe())

#1970 çıkarılıyor
dataFrame = dataFrame[dataFrame.year != 1970]
print(dataFrame.groupby("year").mean()["price"])

#model oluşturma
y = dataFrame["price"].values
x = dataFrame.drop("price",axis=1).values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=10)
print(len(x_train)) #doğru bölünmüş mü kontrol edildi
print(len(x_test))

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

print(x_train.shape)
model = Sequential()
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))

model.add(Dense(1))

model.compile(optimizer="adam",loss="mse")
#model eğitimi
model.fit(x=x_train, y=y_train,validation_data=(x_test,y_test),batch_size=250,epochs=300)

kayipVerisi = pd.DataFrame(model.history.history)
print(kayipVerisi.head())
kayipVerisi.plot()
plt.show()
#sonuçları değerlendirme
from sklearn.metrics import mean_squared_error, mean_absolute_error
tahminDizisi = model.predict(x_test)
print(tahminDizisi)
print(mean_absolute_error(y_test,tahminDizisi)) #veriyi kullanacak kişi karar verecek yeterli de olabilir, olmayabilir de
print(dataFrame.describe())

plt.scatter(y_test,tahminDizisi)
plt.plot(y_test,y_test,"g-")
plt.show()

dataFrame.iloc[2]
yeniArabaSeries = dataFrame.drop("price",axis=1).iloc[2]
print(yeniArabaSeries)
yeniArabaSeries = scaler.transform(yeniArabaSeries.values.reshape(-1,5))
print(model.predict(yeniArabaSeries))



