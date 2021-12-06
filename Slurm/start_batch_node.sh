#!/bin/bash
#SBATCH --job-name=cartoon-gan
#SBATCH --mail-type=ALL
#SBATCH --mail-user=luca.deluigi3@studio.unibo.it
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --output=/public/luca.deluigi3/log/cartoon-gan.log

echo Started...
WD=/public/luca.deluigi3/cartoon-gan
cd "$WD"
source "$WD/bin/activate"
echo "Folder suffix: ${1} Epochs: ${2} Omega: ${3}"
export FOLDER_SUFFIX=${1}
export RESET_CHECKPOINTS=false
python moviefy.py "${2}" "${3}"
