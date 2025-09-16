# settings.py

import os
import json
from utils.path_utils import get_app_data_dir

class Settings:
    def __init__(self, app_name="AIStudio", filename="settings.json"):
        self.app_dir = get_app_data_dir(app_name)
        self.filepath = os.path.join(self.app_dir, filename)
        self.data = self._load_from_disk()

    def _load_from_disk(self):
        try:
            if os.path.exists(self.filepath):
                with open(self.filepath, 'r') as f:
                    return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading settings: {e}")
        return {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def save(self, new_data_dict):
        """Merge new data with existing data and save to disk."""
        self.data.update(new_data_dict)
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.data, f, indent=4)
        except IOError as e:
            print(f"Error saving settings: {e}")
            
    def load(self):
        return self.data