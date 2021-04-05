import tensorflow as tf
import scipy.misc
import numpy as np
from module import *
import os
import secrets


def style_transfer(filepath,stylename):

    filepath = filepath
    batch_size = 1
    image_size = 1280
    to_save_dir = 'output'
    loc='models/'+'model_'+ stylename +'/checkpoint_long'
    print(loc)
    checkpoint_dir = loc

    def normalize_arr_of_imgs(arr):
        return arr/127.5 - 1.

    def denormalize_arr_of_imgs(arr):
        return (arr + 1.) * 127.5

    with tf.name_scope('placeholder'):
        input_photo = tf.placeholder(dtype=tf.float32,
                                     shape=[batch_size, None, None, 3],
                                     name='photo')

        input_photo_features = encoder(image=input_photo,
                                       reuse=False)

        output_photo = decoder(features=input_photo_features,
                               reuse=False)

    saver = tf.train.Saver(max_to_keep=2)
    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        initial_step = int(ckpt_name.split("_")[-1].split(".")[0])
        print("Load checkpoint %s. Initial step: %s." %
              (ckpt_name, initial_step))
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))

    img_path = filepath
    img = scipy.misc.imread(img_path, mode='RGB')
    img_shape = img.shape[:2]

    alpha = float(image_size) / float(min(img_shape))
    img = scipy.misc.imresize(img, size=alpha)
    img = np.expand_dims(img, axis=0)

    img = sess.run(
        output_photo,
        feed_dict={
            input_photo: normalize_arr_of_imgs(img),
        })

    img = img[0]
    img = scipy.misc.imresize(img, size=img_shape)
    img = denormalize_arr_of_imgs(img)
    img_name = secrets.token_hex(16)
    scipy.misc.imsave(os.path.join(
        to_save_dir, img_name + "_stylized.jpg"), img)
