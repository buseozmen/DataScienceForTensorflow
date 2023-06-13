import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import mean_absolute_error

#veriyi incele
dataFrame = pd.read_csv("possum.csv")
print(dataFrame.head(25))
print(dataFrame.describe())
print(dataFrame.isnull().sum()) #footlenght 1 ve age 2 boşluk

#boşlukları ortalama degerle doldur
print(dataFrame.describe()["footlength"])
dataFrame["footlength"].fillna(68.459223, inplace=True)

print(dataFrame.describe()["age"])
dataFrame["age"].fillna(3.833333, inplace=True)

#boşluk kaldı mı kontrol et
print(dataFrame.isnull().sum())

#str ve gereksiz attribute'ları atalım
print(dataFrame["Pop"].unique()) #vic,other
print(dataFrame["sex"].unique()) #f m

def encode(x):
    if x == "Vic":
        return 1
    elif x == "other":
        return 0
dataFrame["EncodePop"]=dataFrame.apply(lambda x: encode(x["Pop"]),axis=1)
def encode(y):
    if y == "f":
        return 1
    elif y == "m":
        return 0
dataFrame["EncodeSex"]=dataFrame.apply(lambda x: encode(x["sex"]),axis=1)

dataFrame.drop("case",axis=1,inplace=True)
dataFrame.drop("Pop",axis=1,inplace=True)
dataFrame.drop("sex",axis=1,inplace=True)

#corr incelmesi
print(dataFrame.corr()["age"])

#grafik incelemeleri
sbn.histplot(dataFrame["age"])
plt.show()
sbn.countplot(x = dataFrame["belly"])
plt.show()
sbn.scatterplot(x="footlength",y="age",data=dataFrame)
plt.show()

#en yüksek en düşük inceleme
print(dataFrame.sort_values("age",ascending = False).head())
print(dataFrame.sort_values("age",ascending = True).head())

#model oluşturma
y = dataFrame["age"].values
x = dataFrame.drop("age",axis=1).values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=10)
print(len(x_train)) #72
print(len(x_test)) #32

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = Sequential()
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))

model.add(Dense(1))

model.compile(optimizer="adam",loss="mse")
earlyStopping = EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=1)

#model eğitimi
model.fit(x=x_train,y=y_train,validation_data=(x_test,y_test),batch_size=1,epochs=120,verbose=1)

kayipVerisi = pd.DataFrame(model.history.history)
print(kayipVerisi.head())
kayipVerisi.plot()
plt.show()

#sonuçları değerlendirme
tahminDizisi = model.predict(x_test)
print(tahminDizisi)
print(mean_absolute_error(y_test,tahminDizisi))

plt.scatter(y_test,tahminDizisi)
plt.plot(y_test,y_test,"g-")
plt.show()

yeniSicanSeries = dataFrame.drop("age",axis=1).iloc[45]
print(yeniSicanSeries)
yeniSicanSeries = scaler.transform(yeniSicanSeries.values.reshape(-1,12))
print(model.predict(yeniSicanSeries))



