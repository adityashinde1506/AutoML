import tensorflow as tf

class Loss:
    
    def mse_loss(self,predictions,labels,samples):
        return tf.divide(tf.reduce_sum(tf.pow(predictions-labels,2)),tf.cast(2.0*samples,tf.float64),name="loss")
