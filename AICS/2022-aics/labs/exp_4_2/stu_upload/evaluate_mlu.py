# coding:utf-8
from __future__ import print_function
import sys
sys.path.insert(0, 'src')
import os
import scipy.misc
import tensorflow as tf
from utils import save_img, get_img, exists, list_files
from argparse import ArgumentParser
from collections import defaultdict
import time
import json
import subprocess
import numpy as np

BATCH_SIZE = 4
DEVICE = '/cpu:0'

os.putenv('MLU_VISIBLE_DEVICES','0')

# get img_shape
def ffwd(data_in, paths_out, model, device_t='', batch_size=1):
    assert len(paths_out) > 0
    is_paths = type(data_in[0]) == str
    # TODO：如果 data_in 是保存输入图像的文件路径，即 is_paths 为 True，则读入第一张图像，由于 pb 模型的输入维度为 1 × 256 × 256 × 3, 因此需将输入图像的形状调整为 256 × 256，并传递给 img_shape；
    # 如果 data_in 是已经读入图像并转化成数组形式的数据，即 is_paths 为 False，则直接获取图像的 shape 特征 img_shape
    if is_paths:
        shape = get_img(data_in[0], (256, 256, 3)).shape
    else:
        shape = data_in[0].shape

    g = tf.Graph()
    # setting mlu configurations
    config = tf.ConfigProto(allow_soft_placement=True,
                    inter_op_parallelism_threads=1,
                    intra_op_parallelism_threads=1)
    config.mlu_options.data_parallelism = 1
    config.mlu_options.model_parallelism = 1
    config.mlu_options.core_num = 16 # 1 4 16
    config.mlu_options.core_version = "MLU270"
    config.mlu_options.precision = "int8"
    config.mlu_options.save_offline_model = False
    with g.as_default():
        with tf.gfile.FastGFile(model,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            input_tensor = sess.graph.get_tensor_by_name('X_content:0')
            output_tensor = sess.graph.get_tensor_by_name('add_37:0')
            batch_size = 1
            # TODO：读入的输入图像的数据格式为 HWC，还需要将其转换成 NHWC
            batch_shape = [batch_size] + list(shape)
            num_iters = int(len(paths_out)/batch_size)
            for i in range(num_iters):
                pos = i * batch_size
                curr_batch_out = paths_out[pos:pos+batch_size]
                # TODO：如果 data_in 是保存输入图像的文件路径，则依次将该批次中输入图像文件路径下的 batch_size 张图像读入数组 X；
                # 如果 data_in 是已经读入图像并转化成数组形式的数据，则将该数组传递给 X
                if is_paths:
                    X = np.empty(batch_shape)
                    for i, j in enumerate(range(pos, pos + batch_size)):
                        X[i] = get_img(data_in[j], shape)
                else:
                    X = data_in[pos: pos + batch_size]
              
                start = time.time()
                # TODO: 使用 sess.run 来计算 output_tensor
                _preds = sess.run(output_tensor, feed_dict={input_tensor: X})
                end = time.time()
                for j, path_out in enumerate(curr_batch_out):
                    #TODO：在该批次下调用 utils.py 中的 save_img() 函数对所有风格迁移后的图片进行存储
                    save_img(path_out, _preds[j])
                delta_time = end - start	
                print("Inference (MLU) processing time: %s" % delta_time)  

def ffwd_to_img(in_path, out_path, model, device='/cpu:0'):
    paths_in, paths_out = [in_path], [out_path]
    ffwd(paths_in, paths_out, model, batch_size=1, device_t=device)

def ffwd_different_dimensions(in_path, out_path, model, 
            device_t=DEVICE, batch_size=4):
    in_path_of_shape = defaultdict(list)
    out_path_of_shape = defaultdict(list)
    for i in range(len(in_path)):
        in_image = in_path[i]
        out_image = out_path[i]
        shape = "%dx%dx%d" % get_img(in_image).shape
        in_path_of_shape[shape].append(in_image)
        out_path_of_shape[shape].append(out_image)
    for shape in in_path_of_shape:
        print('Processing images of shape %s' % shape)
        ffwd(in_path_of_shape[shape], out_path_of_shape[shape], 
            model, device_t, batch_size)

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str,
                        dest='model',
                        help='dir or .pb file to load model',
                        metavar='MODEL', required=True)  

    parser.add_argument('--in-path', type=str,
                        dest='in_path',help='dir or file to transform',
                        metavar='IN_PATH', required=True)

    help_out = 'destination (dir or file) of transformed file or files'
    parser.add_argument('--out-path', type=str,
                        dest='out_path', help=help_out, metavar='OUT_PATH',
                        required=True)

    parser.add_argument('--device', type=str,
                        dest='device',help='device to perform compute on',
                        metavar='DEVICE', default=DEVICE)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size',help='batch size for feedforwarding',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--allow-different-dimensions', action='store_true',
                        dest='allow_different_dimensions', 
                        help='allow different image dimensions')

    return parser

def check_opts(opts):
    exists(opts.model, 'Model not found!')
    exists(opts.in_path, 'In path not found!')
    if os.path.isdir(opts.out_path):
        exists(opts.out_path, 'out dir not found!')
        assert opts.batch_size > 0

def main():
    parser = build_parser()
    opts = parser.parse_args()
    check_opts(opts)

    if not os.path.isdir(opts.in_path):
        if os.path.exists(opts.out_path) and os.path.isdir(opts.out_path):
            out_path = os.path.join(opts.out_path,os.path.basename(opts.in_path))
        else:
            out_path = opts.out_path

        ffwd_to_img(opts.in_path, out_path, opts.model,
                    device=opts.device)
    else:
        files = list_files(opts.in_path)
        full_in = [os.path.join(opts.in_path,x) for x in files]
        full_out = [os.path.join(opts.out_path,x) for x in files]
        if opts.allow_different_dimensions:
            ffwd_different_dimensions(full_in, full_out, opts.model, 
                    device_t=opts.device, batch_size=opts.batch_size)
        else :
            ffwd(full_in, full_out, opts.model, device_t=opts.device,
                    batch_size=opts.batch_size)

if __name__ == '__main__':
    main()
