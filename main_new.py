from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1
logs_path = '/tmp/tensorflow_logs/example'
x = tf.placeholder("float", shape=[None, 784])
x_image = tf.reshape(x, [-1,28,28,1])
y_ = tf.placeholder("float", shape=[None, 10],name='LabelData')

# Set model weights
#W = tf.Variable(tf.zeros([784, 10]), name='Weights')
#b = tf.Variable(tf.zeros([10]), name='Bias')
#---------------------------------------------------
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

#-------------------------------------------------
with tf.name_scope('Model'):
	 pred=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
with tf.name_scope('Loss'):
     cost = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(pred), reduction_indices=1))
	
with tf.name_scope('SGD'):
     optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
	
with tf.name_scope('Accuracy'):
     acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
     acc = tf.reduce_mean(tf.cast(acc, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Create a summary to monitor cost tensor
tf.scalar_summary("loss", cost)
# Create a summary to monitor accuracy tensor
tf.scalar_summary("accuracy", acc)
# Merge all summaries into a single op
merged_summary_op = tf.merge_all_summaries()

with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c, summary = sess.run([optimizer, cost, merged_summary_op],
                                     feed_dict={x: batch_xs, y_: batch_ys})
            summary_writer.add_summary(summary, epoch * total_batch + i)
            avg_cost += c / total_batch
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    print("Optimization Finished!")
    print("Accuracy:", acc.eval({x: mnist.test.images, y_: mnist.test.labels}))
    print("Run the command line:\n"           "--> tensorboard --logdir=/tmp/tensorflow_logs "           "\nThen open http://0.0.0.0:6006/ into your web browser")

