"""
Adapted from https://github.com/google/jax/discussions/9582.
Original author: zhangqiaorjc@google.com

`jax.distributed.initialize` is available in jax-0.2.25.
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html  # Note: wheels only available on linux.

Run this script on 2 GPU nodes, assuming 10.128.0.6 is the master node
python nvidia_gpu_pjit.py --server_addr="10.128.0.6:1456" --num_hosts=2 --host_idx=0
python nvidia_gpu_pjit.py --server_addr="10.128.0.6:1456" --num_hosts=2 --host_idx=1
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from jax.experimental import maps
from jax.experimental.pjit import PartitionSpec as P
from jax.experimental.pjit import pjit

flags.DEFINE_string("server_addr", "", help="server ip addr")
flags.DEFINE_integer("num_hosts", 1, help="num of hosts")
flags.DEFINE_integer("host_idx", 0, help="index of current host")
FLAGS = flags.FLAGS


def main(argv):
    if "SLURM_PROCID" in os.environ:  # for slurm scheduler
        FLAGS.host_idx = int(os.environ["SLURM_PROCID"])
    print(FLAGS.host_idx, "hi!")

    jax.distributed.initialize(
        FLAGS.server_addr,
        FLAGS.num_hosts,
        FLAGS.host_idx,
    )
    print(FLAGS.host_idx, "hi again!")
    print("global devices=", jax.devices())
    print("local devices=", jax.local_devices())

    def f(x, w):
        return jnp.einsum("blm,md->bld", x, w)

    x = jnp.ones((2, 4, 20))
    w = jnp.ones((20, 4))
    print(f(x, w).shape)

    # Model parallelism via pjit
    n = jax.device_count()
    mesh_shape = (n,)
    device_mesh = np.array(jax.devices()).reshape(mesh_shape)
    with maps.Mesh(device_mesh, ("mdl",)):
        result = pjit(
            f,
            in_axis_resources=(P(None, None, "mdl"), P("mdl", None)),
            out_axis_resources=None,
        )(x, w)
    print(result)

    # result is replicated on each chip
    print("print shapes of result on each chip locally")
    for i in range(len(result.device_buffers)):
        print(result.device_buffers[i].shape)


if __name__ == "__main__":
    app.run(main)
