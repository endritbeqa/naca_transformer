import os
import functools
from typing import Optional
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding



def fun1():
    if len(jax.local_devices())< 4:
        raise Exception("Notebook requires 8 devices to run") 
    
    sharding = PositionalSharding(mesh_utils.create_device_mesh((4,))) 

    x = jax.random.normal(jax.random.PRNGKey(0), (8192,8192))
    y = jax.device_put(x, sharding.reshape(4,1))
    jax.debug.visualize_array_sharding(y)

    z= jnp.sin(y)
    jax.debug.visualize_array_sharding(z)


def fun2():
    
    x = jax.random.normal(jax.random.PRNGKey(0), (8192,8192))

    devices = mesh_utils.create_device_mesh((4,)) 
    sharding = PositionalSharding(devices)

    x = jax.device_put(x, sharding.reshape(4,1))
    jax.debug.visualize_array_sharding(x) 
    print(sharding)


if __name__== '__main__':
    fun2()