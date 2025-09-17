import runpod
import time
import logging
import re
import json

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class RunPodManager:
    def __init__(self, api_key: str):
        # Validate API key format
        if not isinstance(api_key, str) or not api_key.strip():
            logging.error("Invalid API key: Empty or not a string")
            raise ValueError("API key cannot be empty or non-string")

        cleaned_api_key = api_key.strip().replace('\n', '').replace('\r', '')
        if not re.match(r'^[A-Za-z0-9\-_]{20,100}$', cleaned_api_key):
            logging.error("Invalid API key format")
            raise ValueError("Invalid API key format. Ensure it is 20-100 alphanumeric characters with hyphens/underscores only.")

        try:
            runpod.api_key = cleaned_api_key
            endpoints = runpod.get_endpoints()
            logging.info(f"RunPod initialized successfully. Found {len(endpoints)} endpoints.")
        except Exception as e:
            logging.error(f"Failed to initialize RunPod API: {str(e)}")
            raise RuntimeError(f"Invalid RunPod API key or service unavailable: {str(e)}")

    def get_existing_endpoint(self, name):
        try:
            endpoints = runpod.get_endpoints()
            logging.debug(f"Fetched endpoints: {json.dumps(endpoints, indent=2)}")
            for endpoint in endpoints:
                if endpoint['name'] == name:
                    logging.info(f"Found existing endpoint '{name}' with ID {endpoint['id']}")
                    return endpoint['id']
            return None
        except Exception as e:
            logging.error(f"Failed to get endpoints: {str(e)}")
            return None

    def create_template(self, name, image_name, port):
        import datetime
        try:
            unique_suffix = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
            unique_name = f"{name}-template-{unique_suffix}"
            logging.info(f"Creating template for {name} with image '{image_name}' on port {port}")
            new_template = runpod.create_template(
                name=unique_name,
                image_name=image_name,
                container_disk_in_gb=15,
                ports=str(port),       
                is_serverless=True
            )
            logging.info(f"Created template {new_template['id']} for {name}")
            return new_template['id']
        except Exception as e:
            logging.error(f"Failed to create template {name}: {str(e)}")
            raise

    def create_endpoint(self, name, template_id):
        try:
            gpus = runpod.get_gpus()
            logging.debug(f"Fetched GPUs: {gpus}")
            if not gpus:
                logging.warning("No GPUs found, falling back to default GPU configuration.")
                gpus = [None]
        except Exception as e:
            logging.error(f"Failed to fetch GPUs: {str(e)}. Falling back to default GPU configuration.")
            gpus = [None]

        errors = []
        for idx, gpu in enumerate(gpus):
            try:
                gpu_id = getattr(gpu, 'id', None) if gpu else None
                logging.debug(f"Trying GPU index {idx}: {gpu}")
                payload = {
                    "name": name,
                    "template_id": template_id,
                    "gpu_ids": [gpu_id] if gpu_id else [],
                    "workers_min": 0,
                    "workers_max": 1
                }
                logging.info(f"Attempting to create endpoint '{name}' with GPU {gpu}")
                logging.debug(f"Payload: {json.dumps(payload, indent=2)}")

                new_endpoint = runpod.create_endpoint(**payload)
                logging.info(f"Created endpoint {new_endpoint['id']} with GPU {gpu}")
                return new_endpoint['id']
            except Exception as e:
                logging.warning(f"GPU {gpu} unavailable for {name}: {str(e)}")
                errors.append({"gpu": str(gpu), "error": str(e)})
                continue

        logging.error(f"All GPU attempts failed for {name}. Errors: {json.dumps(errors, indent=2)}")
        raise RuntimeError(f"No available GPUs for {name}. Check credits or wait and retry later. See logs for details.")

    def poll_endpoint(self, endpoint_id, max_timeout=300, poll_interval=10):
        logging.info(f"Polling endpoint {endpoint_id} for readiness (timeout {max_timeout}s)")
        start_time = time.time()
        endpoint = runpod.Endpoint(endpoint_id)

        while time.time() - start_time < max_timeout:
            try:
                health = endpoint.health()
                logging.debug(f"Health for {endpoint_id}: {json.dumps(health, indent=2)}")

                workers = health.get("workers", {})
                ready_count = workers.get("ready", 0)

                if ready_count > 0:
                    logging.info(f"Endpoint {endpoint_id} is ready with {ready_count} worker(s).")
                    return endpoint  # return the Endpoint object itself
            except Exception as e:
                logging.error(f"Error while polling endpoint {endpoint_id}: {e}")

            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, 30)

        raise TimeoutError(f"Endpoint {endpoint_id} not ready after {max_timeout} seconds.")

    def deploy_and_poll_endpoint(self, endpoint_name, image_name, port):
        endpoint_id = self.get_existing_endpoint(endpoint_name)
        if not endpoint_id:
            template_id = self.create_template(endpoint_name, image_name, port)
            endpoint_id = self.create_endpoint(endpoint_name, template_id)

        endpoint = self.poll_endpoint(endpoint_id)  
        return endpoint, endpoint_id

    def terminate_endpoints(self, endpoint_ids):
        for endpoint_id in endpoint_ids:
            try:
                runpod.delete_endpoint(endpoint_id)
                logging.info(f"Terminated endpoint {endpoint_id}")
            except Exception as e:
                logging.error(f"Failed to terminate endpoint {endpoint_id}: {str(e)}")
