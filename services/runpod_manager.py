import requests
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

RUNPOD_API_BASE = "https://api.runpod.io/v2"

class RunPodManager:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def _get_endpoint_by_name(self, name):
        try:
            response = requests.get(f"{RUNPOD_API_BASE}/endpoints", headers=self.headers)
            response.raise_for_status()
            endpoints = response.json()
            for endpoint in endpoints:
                if endpoint['name'] == name:
                    return endpoint
            return None
        except Exception as e:
            logging.error(f"Could not list endpoints: {str(e)}")
            return None

    def deploy_and_poll_endpoint(self, name, image, port, gpu_id="NVIDIA RTX 3090", max_retries=3, timeout_seconds=600):
        endpoint_id = None
        for attempt in range(max_retries):
            try:
                existing_endpoint = self._get_endpoint_by_name(name)
                if existing_endpoint:
                    logging.info(f"Found existing endpoint '{name}' with ID: {existing_endpoint['id']}")
                    endpoint_id = existing_endpoint['id']
                else:
                    logging.info(f"No existing endpoint found. Creating new endpoint '{name}'...")
                    template_payload = {
                        "name": f"{name}-template",
                        "imageName": image,
                        "containerDiskInGb": 15,
                        "ports": f"{port}/http",
                        "isServerless": True
                    }
                    response = requests.post(f"{RUNPOD_API_BASE}/templates", headers=self.headers, json=template_payload)
                    response.raise_for_status()
                    template_id = response.json()['id']
                    logging.info(f"Template '{name}-template' created with ID: {template_id}")
                    endpoint_payload = {
                        "name": name,
                        "templateId": template_id,
                        "gpuIds": gpu_id,
                        "scaler": {"min": 0, "max": 1}
                    }
                    response = requests.post(f"{RUNPOD_API_BASE}/endpoints", headers=self.headers, json=endpoint_payload)
                    response.raise_for_status()
                    endpoint_id = response.json()['id']
                    logging.info(f"Endpoint '{name}' created with ID: {endpoint_id}")

                start_time = time.time()
                backoff = 10.0
                while time.time() - start_time < timeout_seconds:
                    logging.info(f"Polling status for endpoint '{name}' (attempt {attempt+1}/{max_retries})...")
                    response = requests.get(f"{RUNPOD_API_BASE}/endpoints/{endpoint_id}", headers=self.headers)
                    if response.status_code != 200:
                        time.sleep(backoff)
                        backoff = min(backoff * 1.5, 30.0)
                        continue

                    endpoint_status = response.json()
                    workers = endpoint_status.get("workers", [])
                    
                    if workers and workers[0]['status'] == "READY":
                        url = workers[0].get("publicUrl")
                        if url:
                            logging.info(f"Endpoint '{name}' is READY. URL: {url}")
                            return url, endpoint_id
                    time.sleep(backoff)
                    backoff = min(backoff * 1.5, 30.0)
                
                logging.warning(f"Endpoint '{name}' did not become ready within {timeout_seconds} seconds.")
            except Exception as e:
                logging.error(f"Deployment attempt {attempt+1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(5)
        raise TimeoutError(f"Failed to deploy endpoint '{name}' after {max_retries} attempts.")

    def terminate_endpoints(self, endpoint_ids):
        if not endpoint_ids:
            return

        logging.info(f"Terminating RunPod endpoints: {endpoint_ids}")
        for endpoint_id in endpoint_ids:
            if not endpoint_id:
                continue
            try:
                response = requests.delete(f"{RUNPOD_API_BASE}/endpoints/{endpoint_id}", headers=self.headers)
                if response.status_code == 200:
                    logging.info(f"Successfully terminated endpoint {endpoint_id}.")
                else:
                    logging.warning(f"Could not terminate endpoint {endpoint_id}. Status: {response.status_code}, Response: {response.text}")
            except Exception as e:
                logging.error(f"Error terminating endpoint {endpoint_id}: {str(e)}")