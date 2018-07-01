# _*_ coding:utf-8 _*_
__author__ = 'jiangchao'
__date__ = '2018/7/1 0001 下午 4:19'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot
import random

from captcha.image import ImageCaptcha
from PIL import Image

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
            'u', 'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z']


def random_captcha_text(char_set=number + alphabet + ALPHABET, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


def generate_captcha_text_and_image():
    image = ImageCaptcha()
    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)

    captcha = image.generate(captcha_text)

    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image


def convert_2_gray(image):
    if len(image.shape) > 2:
        gray = np.mean(image, -1)
        return gray
    else:
        return image


IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
# text, image = generate_captcha_text_and_image()
# MAX_CAPTCHA = len(text)
MAX_CAPTCHA = 4
char_set = number + alphabet + ALPHABET
CHAR_SET_LEN = len(char_set)


def text_2_vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

    def char2pos(c):  
        if c =='_':  
            k = 62  
            return k  
        k = ord(c)-48  
        if k > 9:  
            k = ord(c) - 55  
            if k > 35:  
                k = ord(c) - 61  
                if k > 61:  
                    raise ValueError('No Map')   
        return k  

    for i, c in enumerate(text):
        idx = char2pos(c)
        vector[idx] = 1
    return vector


def vec2text(vec):

    char_pos = vec.nonzero()[0]
    text=[]
    for i, c in enumerate(char_pos):
        char_at_pos = i
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx <36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx-  36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    # text = []
    # char_pos = vec.nonzero()[0]
    # for i, c in enumerate(char_pos):
    #     number = i % 10
    #     text.append(str(number))

    return "".join(text)


def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = generate_captcha_text_and_image()
            if image.shape == (60, 160, 3):
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert_2_gray(image)

        batch_x[i, :] = image.flatten() / 255
        batch_y[i, :] = text_2_vec(text)
    return batch_x, batch_y


if __name__ == '__main__':
    text, image = generate_captcha_text_and_image()
    f = plot.figure()
    ax = f.add_subplot(111)
    ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
    plot.imshow(image)
    plot.show()
