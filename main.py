import pandas as pd
import catboost as cb
from sklearn.model_selection import train_test_split
import numpy as np

data  = pd.read_csv("1026izi.txt")
print (data)
train_label = data['result']
del data['result']
train_data = data.head(400)
test_data = data[400:-1]

train_pool = cb.Pool(train_data,
                  train_label,
                  cat_features=[0,2,5])
test_pool = cb.Pool(test_data,
                 cat_features=[0,2,5])

# specify the training parameters
model = cb.CatBoostRegressor(iterations=2,
                          depth=2,
                          learning_rate=1,
                          loss_function='RMSE')
#train the model
model.fit(train_pool)
# make the prediction using the resulting model
preds = model.predict(test_pool)
print(preds)
