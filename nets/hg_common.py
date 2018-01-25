# coding:utf8
import tensorflow.contrib.slim as slim
import tensorflow as tf


def bottle_neck(inputs,
                planes,
                stride=1,
                downsample=None,
                is_training=True,
                reuse=None,
                scope="bottle_neck"):
    net = inputs
    residual = inputs
    with tf.variable_scope(name_or_scope=scope, reuse=reuse):
        net = slim.batch_norm(net, activation_fn=tf.nn.relu, is_training=is_training, scope='bn1')
        net = slim.conv2d(net, planes, kernel_size=[1, 1], padding='VALID', scope='conv1')

        net = slim.batch_norm(net, activation_fn=tf.nn.relu, is_training=is_training, scope='bn2')
        net = slim.conv2d(net, planes, kernel_size=[3, 3], stride=stride, padding='SAME', scope='conv2')

        net = slim.batch_norm(net, activation_fn=tf.nn.relu, is_training=is_training, scope='bn3')
        net = slim.conv2d(net, planes * 2, kernel_size=[1, 1], padding='VALID', scope='conv3')

        if downsample:
            residual = downsample(inputs)
        net += residual
        return net


class DownSample(object):
    def __init__(self, planes, kernel_size=1, stride=1, reuse=None, scope='down'):
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.reuse = reuse
        self.scope = scope

    def __call__(self, inputs):
        net = inputs
        with tf.variable_scope(name_or_scope=self.scope):
            net = slim.conv2d(net, self.planes, self.kernel_size, self.stride, reuse=self.reuse, scope=self.scope)
            return net


def bottle_neck_stack(inputs, planes, num_blocks=4, is_training=True, reuse=None, scope='res'):
    net = inputs
    with tf.variable_scope(name_or_scope=scope):
        for i in range(num_blocks):
            net = bottle_neck(net, planes, is_training=is_training, reuse=reuse, scope='btn_%d' % i)
        return net


class Hourglass(object):
    def __init__(self, num_blocks=4, planes=128, depth=4):
        """
        沙漏网络：一个hg一般有4个res（res0,res1,res2,res3），res0有4个[bottle_neck0,bottle_neck1,bottle_neck2,bottle_neck3],
        其他res都只有三个[bottle_neck0,bottle_neck1,bottle_neck2,bottle_neck3]
        :param num_blocks: 每个res里有几个bottle_neck
        :param planes: 中间的channel变化：in_channels -> planes -> planes -> planes*2
        :param depth: hg的深度，一般指depth个res，也是递归的深度
        """
        self.num_blocks = num_blocks
        self.planes = planes
        self.depth = depth

    def __call__(self, inputs, is_training=True, reuse=None, scope='hourglass'):
        return self.hourglass(inputs, is_training, reuse, scope)

    def hourglass(self, inputs, is_training=True, reuse=None, scope='hourglass'):
        with tf.variable_scope(name_or_scope=scope) as scope:
            hg_out = self._hg_forward(inputs, self.planes, self.depth, 0, is_training, scope, reuse)
            return hg_out

    def _hg_forward(self, inputs, planes, depth, hgid=0, is_training=True, parent_scope='hg', reuse=None):
        up1 = inputs
        with tf.variable_scope(name_or_scope=parent_scope, default_name='hourglass'):
            with tf.variable_scope('res_%d' % hgid):
                up1 = bottle_neck_stack(up1, planes, self.num_blocks, is_training=is_training, reuse=reuse, scope='up1')
                pool = slim.max_pool2d(up1, kernel_size=2, stride=2, scope='pool')
                low1 = bottle_neck_stack(pool, planes, self.num_blocks, is_training=is_training, reuse=reuse,
                                         scope='low1')
                if depth > 1:
                    low2 = self._hg_forward(low1, planes, depth - 1, hgid + 1, is_training, parent_scope, reuse)
                else:
                    low2 = bottle_neck_stack(low1, planes, self.num_blocks, is_training=is_training, reuse=reuse,
                                             scope='low2')
                low3 = bottle_neck_stack(low2, planes, self.num_blocks, is_training=is_training, reuse=reuse,
                                         scope='low3')
                up2 = tf.image.resize_nearest_neighbor(low3, tf.shape(low3)[1:3] * 2, name='up2')
                return tf.add_n([up2, up1], name='out_hg_%d' % hgid)

    def build(self):
        inputs = tf.placeholder(tf.float32, [8, 64, 64, 256], name='inputs')
        out = self.hourglass(inputs)
        return out


def conv_bn_relu(inputs, planes, is_training=True, reuse=None, scope='c_b_r'):
    net = inputs
    with tf.variable_scope(name_or_scope=scope, reuse=reuse):
        net = slim.conv2d(net, planes, kernel_size=1, scope='conv')
        net = slim.batch_norm(net, activation_fn=tf.nn.relu, is_training=is_training, scope='bn')
        return net


def make_residual(inputs, planes, num_blocks, is_training=True, reuse=None, scope='res'):
    net = inputs
    with tf.variable_scope(name_or_scope=scope):
        for i in range(num_blocks):
            net = bottle_neck(net, planes, is_training=is_training, reuse=reuse, scope='bottle_%d' % i)
        return net


def main():
    # inputs = tf.placeholder(tf.float32, [8, 64, 64, 256], name='inputs')
    # out = bottle_neck(inputs, 128)
    hg = Hourglass()
    out = hg.build()
    print('out-------:', out)
    vars = slim.get_model_variables()
    for i, v in enumerate(vars):
        print(i, '---', v)


if __name__ == '__main__':
    main()
