#!/bin/bash
mkdir -p log
virtualenv -p python3 cartoon-gan
cd cartoon-gan
source ./bin/activate
pip install --no-cache-dir -r requirements.txt
jupyter nbextension enable --py widgetsnbextension
cd ..

#sbatch start_jupyter_node.sh
