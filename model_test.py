
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

model = tf.keras.models.load_model('models/ANNv3.h5')

x_test = pd.read_csv('drive_cycle_data/X_TEST.csv')
y_test = pd.read_csv('drive_cycle_data/Y_TEST.csv')

res = model.predict(np.array(x_test))

plt.figure(1)
plt.title('estimation')
plt.plot(y_test, label='y', c='b')
plt.plot(res, label='est', c='r')
plt.legend()
plt.show()