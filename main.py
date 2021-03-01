import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

if __name__ == '__main__':
    print('TF version: ', tf.__version__)
    print('Num GPUs available: ', len(tf.config.list_physical_devices('GPU')))
    print('Is build with CUDA: ', tf.test.is_built_with_cuda())
    print('Is build with GPU support: ', tf.test.is_built_with_gpu_support())
