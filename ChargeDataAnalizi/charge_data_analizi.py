import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import sklearn.preprocessing as skprp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import mean_absolute_error,mean_squared_error

#veriyi incele
dataFrame = pd.read_csv("sigorta.csv")
print(dataFrame.head())
print(dataFrame.describe())
print(dataFrame.isnull().sum()) #null değer yok

#str değerli sütunları encode ile int e çevir
def encode(x):
    if x == "yes":
        return 1
    elif x =="no":
        return 0
dataFrame["EncodeSmoker"]=dataFrame.apply(lambda x: encode(x["smoker"]),axis=1)

def encode(y):
    if y == "southwest":
        return 1
    elif y =="southeast":
        return 2
    elif y =="northwest":
        return 3
    elif y =="northeast":
        return 4
dataFrame["EncodeRegion"]=dataFrame.apply(lambda y: encode(y["region"]),axis=1)

def encode(z):
    if z == "female":
        return 1
    elif z =="male":
        return 2
dataFrame["EncodeSex"]=dataFrame.apply(lambda z: encode(z["sex"]),axis=1)

#str ve gereksiz sütunları sil
dataFrame.drop("smoker",axis=1,inplace=True)
dataFrame.drop("sex",axis=1,inplace=True)
dataFrame.drop("region",axis=1,inplace=True)

#verinin düzenlenmiş halini kontrol et
print(dataFrame.head())
print(dataFrame.corr()["charges"])

#grafik incelemesi
sbn.histplot(dataFrame["charges"]) #displot
plt.show()
sbn.countplot(x = dataFrame["age"])
plt.show()
sbn.scatterplot(x="bmi",y="charges",data=dataFrame)
plt.show()

#en yüksek en düşük değerleri inceleme
print(dataFrame.sort_values("charges",ascending = False).head())
print(dataFrame.sort_values("charges",ascending = True).head())

#model oluşturma
y = dataFrame["charges"].values
x = dataFrame.drop("charges",axis=1).values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=10)
print(len(x_train))
print(len(x_test))

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

#model eğitimi
model.fit(x=x_train,y=y_train,validation_data=(x_test,y_test),batch_size=1,verbose=1,epochs=400)

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

yeniChargesSeries = dataFrame.drop("charges",axis=1).iloc[2]
print(yeniChargesSeries)
yeniChargesSeries = scaler.transform(yeniChargesSeries.values.reshape(-1,6))
print(model.predict(yeniChargesSeries))
