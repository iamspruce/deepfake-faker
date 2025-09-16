import sys

def check_gpu():
    """
    Safely checks for a compatible NVIDIA GPU using PyTorch.
    Returns False if torch is not installed or fails to import,
    preventing the entire application from crashing.
    """
    try:
        # We only import torch inside this function to avoid
        # it being a hard dependency for the entire app.
        import torch
        
        if torch.cuda.is_available():
            print(f"NVIDIA GPU found: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("No compatible NVIDIA GPU found.")
            return False
    except ImportError:
        print("PyTorch is not installed. GPU check skipped. Assuming no GPU.")
        return False
    except Exception as e:
        # Catch other potential torch initialization errors (like the numpy issue)
        print(f"An error occurred during PyTorch import for GPU check: {e}")
        print("Assuming no GPU.")
        return False

