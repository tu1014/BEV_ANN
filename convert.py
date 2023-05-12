
import tensorflow as tf

model = tf.keras.models.load_model('./models/ANNv4.h5')
model.save('./models/MATLAB/ANNv4')