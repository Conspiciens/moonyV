import tensorflow as tf 

# How a layer is created in tensorflow 
# 
# class MyDenseLayer(tf.keras.layers.Layer): 
# 	def __init__(self, input_dim, output_dim): 
# 		super(MyDenseLayer, self).__init__() 
# 
# 		self.W = self.add_weight([input_dim, output_dim]) 
# 		self.b = self.add_weight([1, output_dim]) 
# 
# 	def call(self, inputs): 
# 		# Foward propagate the inputs 
# 		z = tf.matmul(inputs, self.W) + self.b 
# 
# 		output = tf.math.sigmoid(z)
# 
# 		return output 
# 
# Function 
# layer = tf.keras.layer.Dense(units = 2)
# 
# Stack layers (Sequentail Modles)
#	mdoel = tf.keras.Sequential([tf.keras.layers.Dense(n), tf.keras.layers.Dense(2)]) 