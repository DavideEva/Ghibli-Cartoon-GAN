# Slurm configuration
1. Copy each file inside this folder to the server folder.
    ```
    .
    ├── cartoon-gan
    |   └── requirements.txt
    ├── setup.sh
    └── start_jupyter_node.sh
    ```
1. Run `setup.sh`. You need to have permissions (`chmod +x setup.sh`) and you need python3 with virtualenv installed.
1. Copy your notebook files inside the `cartoon-gan` folder
1. Run `sbatch start_jupyter_node.sh` to schedule the script on a slurm node