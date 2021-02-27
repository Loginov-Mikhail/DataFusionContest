# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score, accuracy_score
import io
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from tensorflow.keras import Input, Model, models, optimizers
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, concatenate, Flatten, Embedding
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint

SLICE = 1000000

# Чтение датасета
df = pd.read_parquet('data/task1_test_for_user.parquet', columns=['id', 'item_name'])
df['item_name'] = df['item_name'].astype(str)
df['item_name'] = df['item_name'].str.lower()

# Чтение названий колонок результата работы модели
z = pd.read_csv('columns.csv')
dummiesColumnsName = list(z.columns)

# Загрузка словаря токенайзера
with open('tokenizer_item_name.json') as f:
    data = json.load(f)
    tokenizer_goods = tokenizer_from_json(data)

# Загрузка модели
#from tensorflow.keras import Input, Model, regularizers, optimizers, models

#load structure
json_file = open('model_v3.json', "r")
loaded_model_json = json_file.read()
json_file.close()

model2 = models.model_from_json(loaded_model_json)
# load weights
model2.load_weights('model_v3.h5')
print("Done")
model = Model(model2.input, model2.layers[-1].output)
#model.trainable = True

loss='categorical_crossentropy'#binary_crossentropy categorical_crossentropy
optimiser=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
metrics1='accuracy'
model.compile(optimizer=optimiser, 
              loss=loss, 
              metrics=[metrics1])
model.summary()

# Обработка названий товаров по полученному словарю
sequences_goods = tokenizer_goods.texts_to_sequences(df['item_name'])
max_position_len = 10
# Замена неожиданных значений None на 0
for i in range(len(sequences_goods)):
    sequences_goods[i] = [0 if v is None else v for v in sequences_goods[i]]

x1 = pad_sequences(sequences_goods, maxlen=max_position_len)

del sequences_goods

zFullPred = pd.DataFrame(columns=['pred'])
while len(x1) > SLICE:
    x = x1[:SLICE]
    y = model.predict(x)
    yPred = pd.DataFrame(y, columns=dummiesColumnsName)
    zPred = pd.DataFrame(pd.get_dummies(yPred).idxmax(1),
                         columns=['pred'])
    zPred['pred'] = zPred['pred'].astype('uint8')
    zFullPred = zFullPred.append(zPred)
    x1 = x1[SLICE:]
x = x1
y = model.predict(x)
yPred = pd.DataFrame(y, columns=dummiesColumnsName)
zPred = pd.DataFrame(pd.get_dummies(yPred).idxmax(1), columns=['pred'])
zPred['pred'] = zPred['pred'].astype('uint8')
zFullPred = zFullPred.append(zPred)

zFullPred['pred'] = zFullPred['pred'].astype('uint8')
zFullPred['id'] = df['id']

zFullPred[['id', 'pred']].to_csv('answers.csv', index=False)
