import wandb
import yaml
from revised_main import main
# from parallel_main import main
import argparse
from config_loader import args as base_args

def load_sweep_config(config_path="config/sweep_config.yaml"):
    """
    Load the sweep configuration from YAML file.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train_with_sweep():
    """
    Training function that uses W&B sweep configuration.
    """
    # Initialize wandb with sweep configuration
    wandb.init()
    
    # Create a new argparse.Namespace object with base configuration
    sweep_args = argparse.Namespace(**vars(base_args))
    
    # Update arguments with sweep parameters
    sweep_config = wandb.config
    for key, value in sweep_config.items():
        if hasattr(sweep_args, key):
            setattr(sweep_args, key, value)
    
    # Run the main training function with sweep parameters
    try:
        main(sweep_args)
    except Exception as e:
        wandb.alert(
            title="Training failed",
            text=f"Training failed with error: {str(e)}"
        )
        raise e
    finally:
        wandb.finish()

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    # Load sweep configuration
    sweep_configuration = load_sweep_config()
    
    # Initialize the sweep
    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project="TopoPool-2"  # Make sure this matches your project name
    )
    
    # Start the sweep
    wandb.agent(sweep_id,
                function=train_with_sweep,
                # count=15  # Set `count` to limit the number of runs
                )