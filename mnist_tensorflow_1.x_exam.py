import tensorflow as tf
import keras
import pandas

tf.compat.v1.disable_eager_execution()  # Eager execution 비활성화

# Model Parameters
learning_rate = 0.001
num_epochs = 10
batch_size = 100

# Load MNIST dataset using Keras datasets API
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and preprocess data
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# One-hot encode labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Network Parameters
n_input = 784  # MNIST data input (28*28 pixels)
n_hidden1 = 512
n_hidden2 = 256
n_classes = 10  # MNIST total classes (0-9 digits)

# Placeholders for input and output
X = tf.compat.v1.placeholder(tf.float32, [None, n_input])
Y = tf.compat.v1.placeholder(tf.float32, [None, n_classes])

# Weights and biases initialization
weights = {
    'h1': tf.compat.v1.Variable(tf.random.normal([n_input, n_hidden1])),  # 수정된 부분
    'h2': tf.compat.v1.Variable(tf.random.normal([n_hidden1, n_hidden2])),  # 수정된 부분
    'out': tf.compat.v1.Variable(tf.random.normal([n_hidden2, n_classes]))  # 수정된 부분
}

biases = {
    'b1': tf.compat.v1.Variable(tf.random.normal([n_hidden1])),  # 수정된 부분
    'b2': tf.compat.v1.Variable(tf.random.normal([n_hidden2])),  # 수정된 부분
    'out': tf.compat.v1.Variable(tf.random.normal([n_classes]))  # 수정된 부분
}

# Create model
def multilayer_perceptron(x):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))  # 수정된 부분
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize variables
init = tf.compat.v1.global_variables_initializer()
total_batch = x_train.shape[0] // batch_size

# Create a Dataset object for batching & batch size iterator generation
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
next_batch = iterator.get_next()

# Start Training
with tf.compat.v1.Session() as sess:
    sess.run(init)
    sess.run(iterator.initializer)

    for epoch in range(num_epochs):
        avg_loss = 0.
        for i in range(total_batch):
            batch_x, batch_y = sess.run(next_batch)
            # training logic here...
            _, l = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
            avg_loss += l / total_batch

        # Display logs per epoch
        acc = sess.run(accuracy, feed_dict={X: x_test, Y: y_test})
        print("Epoch:", '%02d' % (epoch+1), "Loss:", "{:.4f}".format(avg_loss), "Accuracy:", "{:.4f}".format(acc) + "%")

        # Re-initialize iterator for next epoch (essential!)
        sess.run(iterator.initializer)

    # Calculate accuracy for MNIST test images
    print("Final Test Accuracy:", sess.run(accuracy, feed_dict={X: x_test, Y: y_test}), "%")
    print()
    print("Tensorflow:", tf.__version__)

    data = {
        '이름': ['정연재'],
        '학번': [2315398],
        '학과': ['인공지능공학부']
    }

    print()
    df = pandas.DataFrame(data)
    print(df)
    print()
