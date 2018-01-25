# coding:utf8
import os
import tensorflow as tf
from nets.hourglass_net import HourglassNet
from nets.hg_common import Hourglass
import tensorflow.contrib.slim as slim


def main():
    hg_net = HourglassNet()
    out = hg_net.build()

    tensorboard_dir = './logs'  # 保存目录
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(tensorboard_dir)
        writer.add_graph(sess.graph)

    print('out-------:', out)
    vars = slim.get_model_variables()
    for i, v in enumerate(vars):
        print(i, '---', v)


if __name__ == '__main__':
    main()
