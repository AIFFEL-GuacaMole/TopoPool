import argparse
import yaml

# Create an argument parser
parser = argparse.ArgumentParser(description='test arg parse')

# Load YAML configuration
with open("config/train_config.yaml", 'r') as file:
    train_config = yaml.safe_load(file)

# Extract parser configuration and flatten it
parser_config = train_config['parser_arg']
flat_config = [list(item.values())[0] for item in parser_config]

# Function to directly pass parameters
def call_function_with_config_params(func, config_item_params: dict):
    # Extract `name_or_flags`
    name_or_flags = config_item_params.get('name_or_flags')
    if not name_or_flags:
        raise ValueError(f"Missing 'name_or_flags' in configuration: {config_item_params}")

    # Convert 'type' string to callable if necessary
    processed_params = {}
    for k, v in config_item_params.items():
        if k == 'type' and isinstance(v, str):  # Handle the `type` key specifically
            v = eval(v)  # Convert from string to callable, e.g., "int" -> int
        if k != 'name_or_flags':  # Exclude 'name_or_flags'
            processed_params[k] = v

    # Debugging output
    # print(f"Calling {func.__name__} with name_or_flags={name_or_flags} and params={processed_params}")

    # Pass `name_or_flags` as the first positional argument and other params as kwargs
    func(name_or_flags, **processed_params)

# Iterate over configuration and add arguments
for arg in flat_config:
    call_function_with_config_params(parser.add_argument, arg)

# Parse arguments (for testing purposes)
args = parser.parse_args()

if __name__ == '__main__':
    print("Parsed Arguments:", vars(args))
