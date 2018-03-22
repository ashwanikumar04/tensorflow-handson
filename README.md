
## Some handson on Tensor flow.

## Hello Tensorflow


```python
import tensorflow as tf
hello_constant = tf.constant('Hello Constant')
with tf.Session() as sess:
    output = sess.run(hello_constant)
print(output)
```

    b'Hello Constant'


# Input


```python
x = tf.placeholder(tf.string)
with tf.Session() as sess:
    output=sess.run(x,feed_dict={x:'Hello World'})
print(output)
```

    Hello World



```python
x = tf.placeholder(tf.int32)
with tf.Session() as sess:
    output=sess.run(x,feed_dict={x:123})
print(output)
```

    123



```python


# TODO: Convert the following to TensorFlow:
x = tf.constant(10)
y = tf.constant(2)
z = tf.subtract(tf.divide(x,y),tf.cast(tf.constant(1),tf.float64))

with tf.Session() as sess:
    output=sess.run(z)
print(output)
```

    4.0



```python
node1 = tf.constant(2)
node2 = tf.constant(3)
with tf.Session() as sess:
    node3 = sess.run(node1+node2)
print(node3)
```

    5


## TensorBoard



```python
import tensorflow as tf
a = tf.constant(4,name='node_a')
b = tf.constant(5,name='node_b')
c = tf.multiply(a,b,name='multiply_c')
d = tf.add(a,b,name='add_d')
e = tf.add(c,d,name='add_e')
sess = tf.Session()
output = sess.run(e)
writer = tf.summary.FileWriter('./graph1',sess.graph)
writer.close()
sess.close()

```

## Run Tensorboard
```
tensorboard --logdir=graph1/

```
```
Starting TensorBoard b'47' at http://0.0.0.0:6006
(Press CTRL+C to quit)

```

# Linear Model


```python
import tensorflow as tf
W = tf.Variable([.3],tf.float32)
b = tf.Variable([-.3],tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x+b
init = tf.global_variables_initializer()
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model-y)
loss = tf.reduce_sum(squared_deltas)
optmizer = tf.train.GradientDescentOptimizer(0.01)
train = optmizer.minimize(loss)
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(linear_model,{x:[1,2,3,4]}))
    print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))
    
print('Training')
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})
    print(sess.run([W,b]))
    


```

    [ 0.          0.30000001  0.60000002  0.90000004]
    23.66
    Training
    [array([-0.9999969], dtype=float32), array([ 0.99999082], dtype=float32)]

