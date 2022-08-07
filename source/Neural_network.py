
#
# Supervised Machine Learning Template
#


from sklearn import metrics, datasets
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers

from tensorflow.keras.layers import Dense, Activation, Dropout 
from tensorflow.keras.callbacks import EarlyStopping 

#
# To generate consistant experimental results
#
np.random.seed(0)

###############################################################################
#
# 1. Data preparation
#
###############################################################################
print('\nData preparation...\n')

# dataset = pd.read_csv('Breast_Cancer.csv')
raw_dataset = datasets.load_breast_cancer(as_frame=True)
my_dataset = raw_dataset.frame
# my_dataset.drop(['id','Unnamed: 32'],axis=1,inplace=True) 
# my_dataset['target'] = my_dataset['target'].map({'B':0,'M':1})

X = my_dataset.iloc[:,0:30].values 
y = my_dataset.iloc[:,30].values

sc = StandardScaler() 
X_norm = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 101)

model = keras.Sequential() # model = Sequential() 
model.add(Dense(units=30,activation='relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(units=15,activation='relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(units=1,activation='sigmoid')) 
model.compile(loss='binary_crossentropy',optimizer='rmsprop', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss',mode='min', verbose=1, patience=25)

###############################################################################
#
# 2. Training
#
###############################################################################
print('\nTraining...\n')
model.fit(  x=X_train, 
            y=y_train, 
            epochs=400, 
            batch_size= 64, 
            validation_data=(X_test, y_test), 
            verbose=1, 
            callbacks=[early_stop]
)

###############################################################################
#
# 3. Estimating
#
###############################################################################
print('\nPredicting...\n')


model_loss = pd.DataFrame(model.history.history) 
ax = model_loss[['loss','val_loss']].plot() 
ax.set_xlabel('Epoch')

###############################################################################
#
# 4. Evaluating
#
###############################################################################
print('\Evaluating...\n')

y_pred = model.predict(X_test)
y_pred = np.round(y_pred).astype(int)
cm = confusion_matrix(y_test,y_pred)
print(cm) 
print(classification_report(y_test,y_pred))

# cm = confusion_matrix(y_test, y_pred)
# print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

print(f'\nAccuracy  : {metrics.accuracy_score(y_test,y_pred)}')
print(f'Precision : {metrics.precision_score(y_test, y_pred)}')
print(f'Recall    : {metrics.recall_score(y_test, y_pred)}')