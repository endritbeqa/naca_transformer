import numpy as np
from absl import app
from absl import logging
import jax
import jax.numpy as jnp
import os
import tensorflow as tf
from naca_transformer.code.naca_training.config import get_config
import matplotlib.pyplot as plt
from src.utilities.pressure_preprocesing import *

from src.transformer.fine_tune_input_pipeline import get_data_from_tfds
import tensorflow_datasets as tfds

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # disable INFO and WARNING messages
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"


def func():
    # Hide GPUs from TF. Otherwise, TF might reserve memory and block it for JAX
    tf.config.experimental.set_visible_devices([], 'GPU')

    logging.info('JAX process: %d / %d', jax.process_index(),
                 jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())

    config = get_config()
    if config.trainer=='train':
        ds_train = get_data_from_tfds(config=config, mode='train')
        for step, batch in enumerate(tfds.as_numpy(ds_train)):

            input = batch['encoder']
            target = batch['decoder']
        
            encoder_input = batch['encoder'][0,:,:,:]
            encoder_input = np.squeeze(encoder_input)
            

            target_x = batch['decoder'][0,:,:,0]
            target_x = np.squeeze(target_x)
            target_y = batch['decoder'][0, :, :, 1]
            target_y = np.squeeze(target_y)
            target_z = batch['decoder'][0, :, :, 2]
            target_z = np.squeeze(target_z)

            plt.imshow(target_x)
            plt.show()

            plt.imshow(target_y)
            plt.show()

            plt.imshow(target_z)
            plt.show()


if __name__ == '__main__':
     func()


