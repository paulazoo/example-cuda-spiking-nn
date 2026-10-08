import yaml

def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    return cfg_dict

def save_config(config_dict, filepath):
    with open(filepath, 'w') as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False)
    return 0
