import os
import yaml
import sys

def load_credentials_from_config():
    """Load Coinbase API credentials from trading_config.yaml and set as environment variables"""
    try:
        config_path = os.path.join(os.path.dirname(__file__), 'trading_bot', 'config', 'trading_config.yaml')
        
        if not os.path.exists(config_path):
            print(f"ERROR: Config file not found at {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check if coinbase configuration exists
        if 'coinbase' not in config:
            print("ERROR: No 'coinbase' section found in config file")
            return False
        
        coinbase_config = config['coinbase']
        
        # Set environment variables
        if 'api_key_name' in coinbase_config:
            os.environ["COINBASE_API_KEY"] = coinbase_config['api_key_name']
            print(f"Set COINBASE_API_KEY from config")
        else:
            print("WARNING: No api_key_name found in coinbase config")
        
        if 'private_key' in coinbase_config:
            os.environ["COINBASE_API_SECRET"] = coinbase_config['private_key']
            print(f"Set COINBASE_API_SECRET from config")
        else:
            print("WARNING: No private_key found in coinbase config")
        
        if 'passphrase' in coinbase_config:
            os.environ["COINBASE_PASSPHRASE"] = coinbase_config['passphrase']
            print(f"Set COINBASE_PASSPHRASE from config")
        
        return True
        
    except Exception as e:
        print(f"ERROR loading config: {str(e)}")
        return False

if __name__ == "__main__":
    print("Loading Coinbase API credentials from config...")
    success = load_credentials_from_config()
    
    if success:
        print("\nEnvironment variables set successfully.")
        
        # If a command was provided, execute it with the environment variables set
        if len(sys.argv) > 1:
            import subprocess
            cmd = " ".join(sys.argv[1:])
            print(f"\nExecuting: {cmd}")
            subprocess.run(cmd, shell=True)
    else:
        print("\nFailed to set environment variables.")
