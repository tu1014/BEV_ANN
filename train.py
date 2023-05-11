
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# read csv
x_train = pd.read_csv('drive_cycle_data/X_TRAIN.csv')
y_train = pd.read_csv('drive_cycle_data/Y_TRAIN.csv')

# build model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_dim=3, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mse']
)

# load previously saved model and proceed additional training
#model = tf.keras.models.load_model('models/ANNv3.h5')

model.summary()

# training
history = model.fit(np.array(x_train), np.array(y_train), epochs=300)
model.save('models/ANNv3.h5')

# show result
plt.figure(1)
plt.title('learning curve')
plt.plot(history.history['loss'])
plt.show()