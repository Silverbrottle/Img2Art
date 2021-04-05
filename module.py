import tensorflow as tf
import tensorflow.contrib.slim as slim

def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset


def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d", activation_fn=None):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=activation_fn,
                           weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                           biases_initializer=None)


def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):

    with tf.variable_scope(name):
        input_ = tf.image.resize_images(images=input_,
                                        size=tf.shape(input_)[1:3] * s,
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return conv2d(input_=input_, output_dim=output_dim, ks=ks, s=1, padding='SAME')


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def encoder(image, reuse=True, name="encoder"):
    gf_dim = 32

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        image = instance_norm(input=image,
                              name='g_e0_bn')
        c0 = tf.pad(image, [[0, 0], [15, 15], [15, 15], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_norm(input=conv2d(c0, gf_dim, 3, 1, padding='VALID', name='g_e1_c'),
                                      name='g_e1_bn'))
        c2 = tf.nn.relu(instance_norm(input=conv2d(c1, gf_dim, 3, 2, padding='VALID', name='g_e2_c'),
                                      name='g_e2_bn'))
        c3 = tf.nn.relu(instance_norm(conv2d(c2, gf_dim * 2, 3, 2, padding='VALID', name='g_e3_c'),
                                      name='g_e3_bn'))
        c4 = tf.nn.relu(instance_norm(conv2d(c3, gf_dim * 4, 3, 2, padding='VALID', name='g_e4_c'),
                                      name='g_e4_bn'))
        c5 = tf.nn.relu(instance_norm(conv2d(c4, gf_dim * 8, 3, 2, padding='VALID', name='g_e5_c'),
                                      name='g_e5_bn'))
        return c5


def decoder(features, reuse=True, name="decoder"):
    gf_dim = 32

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
            return y + x

        # Now stack 9 residual blocks
        num_kernels = features.get_shape().as_list()[-1]
        r1 = residule_block(features, num_kernels, name='g_r1')
        r2 = residule_block(r1, num_kernels, name='g_r2')
        r3 = residule_block(r2, num_kernels, name='g_r3')
        r4 = residule_block(r3, num_kernels, name='g_r4')
        r5 = residule_block(r4, num_kernels, name='g_r5')
        r6 = residule_block(r5, num_kernels, name='g_r6')
        r7 = residule_block(r6, num_kernels, name='g_r7')
        r8 = residule_block(r7, num_kernels, name='g_r8')
        r9 = residule_block(r8, num_kernels, name='g_r9')

        # Decode image.
        d1 = deconv2d(r9, gf_dim * 8, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_norm(input=d1,
                                      name='g_d1_bn'))

        d2 = deconv2d(d1, gf_dim * 4, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_norm(input=d2,
                                      name='g_d2_bn'))

        d3 = deconv2d(d2, gf_dim * 2, 3, 2, name='g_d3_dc')
        d3 = tf.nn.relu(instance_norm(input=d3,
                                      name='g_d3_bn'))

        d4 = deconv2d(d3, gf_dim, 3, 2, name='g_d4_dc')
        d4 = tf.nn.relu(instance_norm(input=d4,
                                      name='g_d4_bn'))

        d4 = tf.pad(d4, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.sigmoid(conv2d(d4, 3, 7, 1, padding='VALID', name='g_pred_c'))*2. - 1.
        return pred
    

