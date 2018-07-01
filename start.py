# _*_ coding:utf-8 _*_
__author__ = 'jiangchao'
__date__ = '2018/7/1 0001 下午 8:11'
from train import *
from test import *

if __name__ == '__main__':
    text, image = generate_captcha_text_and_image()
    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
    y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
    keep_prob = tf.placeholder(tf.float32)
    train_crack_captcha_cnn(X, y, keep_prob)
