import requests
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RunPodManager:
    def __init__(self, api_key):
        self.base_url = "https://api.runpod.io/v2"
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        try:
            response = self.session.get(f"{self.base_url}/status")
            response.raise_for_status()
        except requests.HTTPError as e:
            logging.error(f"RunPod API v2 unavailable: {str(e)}. See https://docs.runpod.io.")
            raise RuntimeError("RunPod API v2 is unavailable.")

    def get_existing_endpoint(self, name):
        try:
            response = self.session.get(f"{self.base_url}/endpoints")
            response.raise_for_status()
            endpoints = response.json().get('endpoints', [])
            for endpoint in endpoints:
                if endpoint['name'] == name:
                    return endpoint['id']
            return None
        except Exception as e:
            logging.error(f"Failed to get endpoints: {str(e)}")
            return None

    def create_template(self, name, image_name, port):
        try:
            payload = {
                "name": f"{name}-template",
                "imageName": image_name,
                "containerDiskInGb": 15,
                "ports": f"{port}/http",
                "isServerless": True
            }
            response = self.session.post(f"{self.base_url}/templates", json=payload)
            response.raise_for_status()
            return response.json()['id']
        except requests.HTTPError as e:
            logging.error(f"Failed to create template {name}: {str(e)}")
            raise

    def create_endpoint(self, name, template_id):
        gpu_types = ["NVIDIA RTX 3090", "NVIDIA A100 40GB", "NVIDIA A40"]
        for gpu in gpu_types:
            try:
                payload = {
                    "name": name,
                    "templateId": template_id,
                    "gpuIds": gpu,
                    "scaler": {"min": 0, "max": 1}
                }
                response = self.session.post(f"{self.base_url}/endpoints", json=payload)
                response.raise_for_status()
                return response.json()['id']
            except requests.HTTPError as e:
                logging.warning(f"GPU {gpu} unavailable for {name}: {str(e)}")
                continue
        raise RuntimeError(f"No available GPUs for {name}. Check RunPod credits or try later.")

    def poll_endpoint(self, endpoint_id, max_timeout=600, poll_interval=10):
        start_time = time.time()
        while time.time() - start_time < max_timeout:
            try:
                response = self.session.get(f"{self.base_url}/endpoints/{endpoint_id}")
                response.raise_for_status()
                data = response.json()
                if data.get('workers', []) and data['workers'][0]['status'] == 'READY':
                    return data['workers'][0]['publicUrl']
            except requests.HTTPError:
                pass
            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, 30)
        raise TimeoutError(f"Endpoint {endpoint_id} not ready after {max_timeout} seconds.")

    def deploy_and_poll_endpoint(self, endpoint_name, image_name, port):
        endpoint_id = self.get_existing_endpoint(endpoint_name)
        if not endpoint_id:
            template_id = self.create_template(endpoint_name, image_name, port)
            endpoint_id = self.create_endpoint(endpoint_name, template_id)
        public_url = self.poll_endpoint(endpoint_id)
        return public_url, endpoint_id

    def terminate_endpoints(self, endpoint_ids):
        for endpoint_id in endpoint_ids:
            try:
                response = self.session.delete(f"{self.base_url}/endpoints/{endpoint_id}")
                response.raise_for_status()
                logging.info(f"Terminated endpoint {endpoint_id}")
            except requests.HTTPError as e:
                logging.error(f"Failed to terminate endpoint {endpoint_id}: {str(e)}")