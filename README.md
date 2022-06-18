# slurm-jax
Experiments with running JAX on multiple nodes in a slurm environment.

## basic
Contains a basic example of distributed JAX.

Command to run on a single node:
```bash
python basic/run.py --server_addr="172.31.130.26:1456"
```

Manual commands to run on two nodes:
```bash
# Node 1
python basic/run.py --server_addr="d-7-14-1:1456" --num_hosts=2 --host_idx=0
# Node 2
python basic/run.py --server_addr="d-7-14-1:1456" --num_hosts=2 --host_idx=1
```
The server is started on host 0!

Slurm command to run on 4 nodes:
```bash
cd basic
sbatch job.sh
```

## adversarial_robustness
Contains code from
[deepmind-research/adversarial_robustness](https://github.com/deepmind/deepmind-research/tree/master/adversarial_robustness).
This is used to check that a more advanced distributed setup works.
