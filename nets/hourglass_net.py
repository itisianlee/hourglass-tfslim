# coding:utf8
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

from .hg_common import Hourglass, bottle_neck, DownSample, conv_bn_relu, make_residual


class HourglassNet(object):
    def __init__(self, num_stacks=2, num_classes=14):
        self.num_stacks = num_stacks
        self.num_classes = num_classes

        self.planes = 64
        self.num_feats = 128

    def net(self, inputs, is_training=True, reuse=None, scope='hourglass_net'):
        net = inputs
        with tf.variable_scope(name_or_scope=scope):
            net = self._pre_hg(net, is_training, reuse)
            out = self._hg_stack(net, is_training, reuse)
        return out

    def _pre_hg(self, inputs, is_training=True, reuse=None, scope='pre_hg'):
        net = inputs
        with tf.variable_scope(name_or_scope=scope, reuse=reuse):
            net = slim.conv2d(net, self.planes, kernel_size=[7, 7], stride=2, padding='SAME', scope='conv1')
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, is_training=is_training, scope='bn1')
            # here out shape [bs,128,128,64]
            net = bottle_neck(net, self.planes, 1, DownSample(self.planes * 2, reuse=reuse), is_training, reuse,
                              'layer1')
            net = slim.max_pool2d(net, kernel_size=2, stride=2, padding='VALID', scope='pool1')
            net = bottle_neck(net, self.planes * 2, 1, DownSample(self.planes * 4, reuse=reuse), is_training, reuse,
                              'layer2')
            net = bottle_neck(net, self.num_feats, is_training=is_training, reuse=reuse, scope='layer3')
            # here out shape [bs,64,64,256]
            return net

    def _hg_stack(self, inputs, is_training=True, reuse=None):
        hourglass = Hourglass()
        out = []
        net = inputs
        for i in range(self.num_stacks):
            with tf.variable_scope(name_or_scope='stack_%d' % i, reuse=reuse):
                net = hourglass(net, is_training, reuse, 'hg')
                net = make_residual(net, self.num_feats, 4, is_training, scope='mk_res')
                net = conv_bn_relu(net, 256, is_training, reuse, scope='c_b_r')
                score = slim.conv2d(net, self.num_classes, kernel_size=1, scope='score')
                out.append(score)
                if i < self.num_stacks - 1:
                    fc_ = slim.conv2d(net, 256, kernel_size=1, scope='fc_')
                    score_ = slim.conv2d(score, 256, kernel_size=1, scope='score_')
                    net = tf.add_n([inputs, fc_, score_], name='stack_%d_out_add' % i)
        return out

    def build(self):
        inputs = tf.placeholder(tf.float32, shape=[8, 256, 256, 3], name='input')
        out = self.net(inputs)
        return out


def main():
    hg_net = HourglassNet()
    out = hg_net.build()

    print('out-------:', out)
    vars = slim.get_model_variables()
    for i, v in enumerate(vars):
        print(i, '---', v)


if __name__ == '__main__':
    main()
