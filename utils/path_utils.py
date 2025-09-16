import os
import sys

def get_app_data_dir(app_name):
    """Get the platform-specific application data directory."""
    if sys.platform == 'win32':
        path = os.path.join(os.environ['APPDATA'], app_name)
    elif sys.platform == 'darwin':
        path = os.path.join(os.path.expanduser('~/Library/Application Support'), app_name)
    else: # Linux
        path = os.path.join(os.path.expanduser('~/.local/share'), app_name)
    
    os.makedirs(path, exist_ok=True)
    return path
