import tensorflow as tf

x=tf.placeholder(tf.float64,[None,290])
y_=tf.placeholder(tf.float64)

def compute_predictions(x):
    W=tf.Variable(tf.random_normal([290,1],mean=0.0,stddev=1.0,dtype=tf.float64))
    b=tf.Variable(tf.random_normal([1],dtype=tf.float64))
    return tf.add(tf.matmul(x,W),b)
