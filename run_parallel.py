import subprocess
import os
from termcolor import colored
from multiprocessing import Pool
from time import sleep
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--n_jobs", default="1", type=int)
parser.add_argument("--experiment_name", default="default", type=str)
args = parser.parse_args()

def run_command(command):
    print(colored(' '.join(command), 'yellow'))
    stdout = subprocess.check_output(command)
    print(stdout)

def main():
    exp_name = args.experiment_name
    n_jobs = args.n_jobs
    # creat runs stack
    commands = []
    for run in range(120):
        commands.append([
            f"python3",
            f"run.py",
            f"--experiment_name={exp_name}",
            f"--run_name=run_{run}"
        ])

    with Pool(processes=n_jobs) as p:
        res = p.map(run_command, commands)

if __name__ == "__main__":
    main()
