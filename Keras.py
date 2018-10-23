from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.preprocessing import Imputer
import numpy as np
import pandas as pd
veri = pd.read_csv('breast-cancer-wisconsin.data')
veri.replace('?', -99999, inplace=True)
Yveri = veri.drop(labels='id', axis=1)
imp = Imputer(missing_values=-99999, strategy="mean", axis=0)
Yveri = imp.fit_transform(Yveri)

giris = Yveri[:, 0:8]
cikis = Yveri[:, 9]

model = Sequential()
model.add(Dense(64, input_dim=8))
model.add(Activation('relu'))

model.add(Dense(32))
model.add(Activation('relu'))

model.add(Dense(64))
model.add(Activation('softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(giris, cikis, epochs=10, batch_size=32, validation_split=0.144)

tahmin = np.array([10,10,10,8,6,1,8,9]).reshape(1, 8)
a = model.predict_classes(tahmin)
if a == 2:
    print(a, "İyi Huylu :)")
else:
    print(a, "Kötü Huylu !")

