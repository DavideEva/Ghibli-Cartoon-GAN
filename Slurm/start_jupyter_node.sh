#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --mail-type=ALL
#SBATCH --mail-user=luca.deluigi3@studio.unibo.it
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=10-00:00:00
#SBATCH --output=/public/jupyter/log/cartoon-gan.jupyter.log

echo Started...
WD=/public/jupyter/cartoon-gan
cd "$WD"
hostname -I
cat /etc/hosts
source "$WD/bin/activate"
jupyter notebook --no-browser --port=8888 --NotebookApp.port_retries=0 --ip=0.0.0.0
