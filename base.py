import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


cols = ["flength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class" ]
df = pd.read_csv("magic04.data", names=cols)
df.head()

df["class"] = (df["class"] == "g").astype(int)

for label in cols[:-1]:
    plt.hist(df[df["class"]==1][label],color='blue', label='gamma', alpha=0.7, density=True)
    plt.hist(df[df["class"]==0][label],color='red', label='hadron', alpha=0.7, density=True)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()


#Train, validation, test datasets

train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

def scale_dataset(dataframe, oversample=False):
  X = dataframe[dataframe.columns[:-1]].values
  y = dataframe[dataframe.columns[-1]].values

  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  if oversample:
    ros = RandomOverSampler()
    X, y = ros.fit_resample(X,y)

  data = np.hstack((X, np.reshape(y, (-1, 1))))

  return data, X, y

#print(len(train[train["class"]==1]))
#print(len(train[train["class"]==0]))

train, X_train, y_train = scale_dataset(train, oversample=True)
valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
test, X_test, y_test = scale_dataset(test, oversample=False)

len(y_train)
sum(y_train == 1)
sum(y_train == 0)
# Above divides points equally 

#K Nearest Neighbor

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report 

X_train = np.nan_to_num(X_train)
y_train = np.nan_to_num(y_train)

knn_model = KNeighborsClassifier(n_neighbors=7)
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

print(classification_report(y_test, y_pred))

