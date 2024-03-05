import numpy as np
from absl import app
from absl import logging
import jax
import os
from PIL import Image
import tensorflow as tf
from naca_transformer.code.naca_training.config import get_config
from src.utilities.pressure_preprocesing import *

from naca_transformer.code.fine_tune.fine_tune import get_fine_tune_data_from_tfds
import tensorflow_datasets as tfds

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # disable INFO and WARNING messages
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

def rescale_array(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)

    if min_val == max_val:
        # Handle the case where all values are the same to avoid division by zero
        return np.zeros_like(arr, dtype=np.uint8)

    scaled_arr = 255 * (arr - min_val) / (max_val - min_val)
    rescaled_arr = scaled_arr.astype(np.uint8)

    return rescaled_arr


def func():
    # Hide GPUs from TF. Otherwise, TF might reserve memory and block it for JAX
    tf.config.experimental.set_visible_devices([], 'GPU')

    logging.info('JAX process: %d / %d', jax.process_index(),
                 jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())

    config = get_config()
    os.makedirs('/local/disk1/ebeqa/naca_transformer/outputs/sss')  
    ds_train = get_fine_tune_data_from_tfds(config=config, mode='train')
    for step, batch in enumerate(tfds.as_numpy(ds_train)):

        encoder_input = batch['encoder'][0,:,:,:]
        encoder_input = np.squeeze(encoder_input)
            

        target_x = batch['decoder'][0,:,:,0]
        target_x = np.squeeze(target_x)
        target_y = batch['decoder'][0, :, :, 1]
        target_y = np.squeeze(target_y)
        target_z = batch['decoder'][0, :, :, 2]
        target_z = np.squeeze(target_z)

        print(target_x)


        numpy_array = encoder_input
        normalized_array = rescale_array(numpy_array)
        image = Image.fromarray(normalized_array)
        image.save('/local/disk1/ebeqa/naca_transformer/outputs/sss/'+str(step)+'encoder.png')

        numpy_array = target_x
        normalized_array = rescale_array(numpy_array)
        image = Image.fromarray(normalized_array)
        image.save('/local/disk1/ebeqa/naca_transformer/outputs/sss/'+str(step)+'x.png')

        numpy_array = target_y
        normalized_array = rescale_array(numpy_array)
        image = Image.fromarray(normalized_array)
        image.save('/local/disk1/ebeqa/naca_transformer/outputs/sss/'+str(step)+'y.png')
            
        numpy_array = target_z
        normalized_array = rescale_array(numpy_array)
        image = Image.fromarray(normalized_array)
        image.save('/local/disk1/ebeqa/naca_transformer/outputs/sss/'+str(step)+'z.png')
            



if __name__ == '__main__':
     func()

