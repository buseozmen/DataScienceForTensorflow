import pandas as pd
import numpy as np

dataFrame = pd.read_excel("maliciousornot.xlsx")
print(dataFrame)
print(dataFrame.info())
print(dataFrame.describe())
print(dataFrame.corr()["Type"].sort_values())

import matplotlib.pyplot as plt
import seaborn as sbn

sbn.countplot(x="Type", data=dataFrame)
plt.show()

dataFrame.corr()["Type"].sort_values().plot(kind="bar")
plt.show()

#sınıflandırma modeli
y = dataFrame["Type"].values
x = dataFrame.drop("Type",axis=1).values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=15)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Activation,Dropout
from tensorflow.python.keras.callbacks import EarlyStopping

print(x_train.shape)

#model oluşturalım
model = Sequential()
model.add(Dense(units=30,activation="relu"))
model.add(Dense(units=15,activation="relu"))
model.add(Dense(units=15,activation="relu"))
model.add(Dense(units=1,activation="sigmoid")) #sınıflandırmada kullanılır
model.compile(loss="binary_crossentropy",optimizer="adam")

#training
model.fit(x = x_train, y = y_train, epochs=700, validation_data=(x_test,y_test),verbose=1) #overfitting görmek için epochs 700 dedik
print(model.history.history)

modelKaybi = pd.DataFrame(model.history.history)
modelKaybi.plot()
plt.show() #overfitting görmüş olduk val_loss saçmalıyorsa durdurmak lazım

#early stopping
model = Sequential()
model.add(Dense(units=30,activation="relu"))
model.add(Dense(units=15,activation="relu"))
model.add(Dense(units=15,activation="relu"))
model.add(Dense(units=1,activation="sigmoid")) #sınıflandırmada kullanılır
model.compile(loss="binary_crossentropy",optimizer="adam")
earlyStopping = EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=25)
model.fit(x=x_train,y=y_train, epochs=700, validation_data= (x_test,y_test),verbose=1,callbacks=[earlyStopping])
#700 den önce duruyor tekrar eğitirken
modelKaybi = pd.DataFrame(model.history.history) #grafik daha iyi öncekinden
modelKaybi.plot()
plt.show()

#dropout
model = Sequential()
model.add(Dense(units=30,activation="relu"))
model.add(Dropout(0.6)) #yüzde kaçında rastgele deneme yapılıp kapancak
model.add(Dense(units=15,activation="relu"))
model.add(Dropout(0.6))
model.add(Dense(units=15,activation="relu"))
model.add(Dropout(0.6))
model.add(Dense(units=1,activation="sigmoid")) #sınıflandırmada kullanılır
model.compile(loss="binary_crossentropy",optimizer="adam")
model.fit(x=x_train,y=y_train, epochs=700, validation_data= (x_test,y_test),verbose=1,callbacks=[earlyStopping])

kayipDF = pd.DataFrame(model.history.history) #daha optimize edildi
kayipDF.plot()
plt.show()

tahminlerimiz = model.predict_classes(x_test)
print(tahminlerimiz)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,tahminlerimiz))
print(confusion_matrix(y_test,tahminlerimiz))
