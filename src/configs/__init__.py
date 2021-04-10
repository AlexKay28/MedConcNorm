import os
import yaml
import numpy as np
from copy import deepcopy
from multiprocessing import Pool, Lock, current_process

RUN_NAME = os.environ.get('RUN_NAME', 'default_run_name')

# Define the name of config file
GENERAL_CONFIG = f'configs_to_choose.yml'
CONFIG_FOR_PIPE = f'temp/run_config_{RUN_NAME}.yml'

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(os.path.dirname(__file__), config_name)) as file:
        config = yaml.safe_load(file)
    return config

def dump_config(config_name, config):
    with open(os.path.join(os.path.dirname(__file__), config_name), 'w') as file:
        config = yaml.dump(config, file)

def choose_random_configuration(config_to_choose):
    config_block = deepcopy(config_to_choose)
    for parameter in config_block.keys():
        if isinstance(config_block[parameter], list):
            if parameter == 'pipeline':
                continue
            config_block[parameter] = np.random.choice(config_block[parameter])
        elif isinstance(config_block[parameter], dict):
            config_block[parameter] = choose_random_configuration(config_block[parameter])
    return config_block

def create_random_configuration(configurations):
    print('Create random configuration')
    temp_config = eval(str(choose_random_configuration(configurations)))
    print(f"create config: '{CONFIG_FOR_PIPE}'")
    dump_config(CONFIG_FOR_PIPE, temp_config)

def delete_process_configuration_file():
    local_path = os.path.dirname(__file__)
    config_name = CONFIG_FOR_PIPE
    to_delete = os.path.join(local_path, config_name)
    os.remove(to_delete)

# Load available configurations
AVAILABLE_CONFIGURATIONS = load_config(GENERAL_CONFIG)

# Load configurations for pipe
try:
    CONFIGURATIONS_PIPE = load_config(CONFIG_FOR_PIPE)
except FileNotFoundError:
    create_random_configuration(AVAILABLE_CONFIGURATIONS)
    CONFIGURATIONS_PIPE = load_config(CONFIG_FOR_PIPE)

GENERAL = CONFIGURATIONS_PIPE['general']
PREPROCESSING = CONFIGURATIONS_PIPE['preprocessing']
MODELING = CONFIGURATIONS_PIPE['modeling']
