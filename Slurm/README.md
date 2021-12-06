# Slurm configuration for batch script
1. Copy each file inside this folder to the server folder.
    ```
    .
    ├── cartoon-gan
    |   └── requirements.txt
    ├── setup.sh
    └── start_batch_node.sh
    ```
2. Run `setup.sh`. You need to have permissions (`chmod +x setup.sh`) and you need python3 with virtualenv installed.
3. Copy your notebook files inside the `cartoon-gan` folder
4. Adjust the `start_batchr_node.sh` script with the right folder name for the working directory and the log file
5. Run `sbatch start_batch_node.sh  <suffix> <epochs> <omega>` to schedule the script on a slurm node
6. Wait a few seconds for slurm
7. Run `cat <log/file/path>` to check inputs

# Slurm configuration for jupyter notebook
1. Copy each file inside this folder to the server folder.
    ```
    .
    ├── cartoon-gan
    |   └── requirements.txt
    ├── setup.sh
    └── start_jupyter_node.sh
    ```
2. Run `setup.sh`. You need to have permissions (`chmod +x setup.sh`) and you need python3 with virtualenv installed.
3. Copy your notebook files inside the `cartoon-gan` folder
4. Adjust the `start_jupyter_node.sh` script with the right folder name for the working directory and the log file
5. Run `sbatch start_jupyter_node.sh` to schedule the script on a slurm node
6. Wait a few seconds for slurm
7. Run `cat <log/file/path>` to see the outputs containing the IP and port of the notebook server, as well as the token/password
8. Open IP:port (port should always be 8888) in your browser
