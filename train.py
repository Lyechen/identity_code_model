# _*_ coding:utf-8 _*_
__author__ = 'jiangchao'
__date__ = '2018/7/1 0001 下午 4:45'
from generate_data import *


def crack_captcha_cnn(X, keep_prob, w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    w_c_1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    b_c_1 = tf.Variable(w_alpha * tf.random_normal([32]))
    conv_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c_1, strides=[1, 1, 1, 1], padding='SAME'), b_c_1))
    conv_1 = tf.nn.max_pool(conv_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
    conv_1 = tf.nn.dropout(conv_1, keep_prob=keep_prob)

    w_c_2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c_2 = tf.Variable(w_alpha * tf.random_normal([64]))
    conv_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv_1, w_c_2, strides=[1, 1, 1, 1], padding='SAME'), b_c_2))
    conv_2 = tf.nn.max_pool(conv_2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
    conv_2 = tf.nn.dropout(conv_2, keep_prob=keep_prob)

    w_c_3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c_3 = tf.Variable(w_alpha * tf.random_normal([64]))
    conv_3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv_2, w_c_3, strides=[1, 1, 1, 1], padding='SAME'), b_c_3))
    conv_3 = tf.nn.max_pool(conv_3, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
    conv_3 = tf.nn.dropout(conv_3, keep_prob=keep_prob)

    w_d = tf.Variable(w_alpha * tf.random_normal([8 * 20 * 64, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv_3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob=keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out


def train_crack_captcha_cnn(X, y, keep_prob):
    ouput = crack_captcha_cnn(X, keep_prob)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ouput, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    predict = tf.reshape(ouput, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_predict = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as session:
        init = tf.global_variables_initializer()
        session.run(init)
        step = 0
        while True:
            batch_x, batch_y = get_next_batch(64)
            _, loss_ = session.run([optimizer, loss], feed_dict={X: batch_x, y: batch_y, keep_prob: 0.75})
            print(step, loss_)

            if step % 10 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                acc = session.run(accuracy, feed_dict={X: batch_x_test, y: batch_y_test, keep_prob: 1.})
                print(step, acc)
                if acc > 0.5:
                    saver.save(session, 'data', global_step=step)
            step += 1
