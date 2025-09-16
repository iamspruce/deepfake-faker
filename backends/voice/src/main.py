import os
import sys
import torch
import urllib.parse
import urllib.request
import shutil
import zipfile
from rvc import rvc_infer, load_hubert, get_vc, Config

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'voice_output')

device = "cuda:0" if torch.cuda.is_available() else "cpu"
#is_half = False if device == "cpu" else True
is_half = False

def download_online_model(url, dir_name, overwrite=False):
    try:
        # Parse the URL and extract the filename
        parsed_url = urllib.parse.urlparse(url)
        zip_name = os.path.basename(parsed_url.path)
        
        # Remove any query parameters from the filename
        zip_name = zip_name.split('?')[0]
        
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            if overwrite:
                print(f"[!] Voice model directory {dir_name} already exists. Overwriting...")
                shutil.rmtree(extraction_folder)
            else:
                print(f"[!] Voice model directory {dir_name} already exists. Using existing model.")
                return f"[+] Using existing model: {dir_name}"

        # Download the file
        print(f"[*] Downloading model from {url}...")
        urllib.request.urlretrieve(url, zip_name)
        
        # Extract the zip file
        print(f"[*] Extracting model to {extraction_folder}...")
        with zipfile.ZipFile(zip_name, 'r') as zip_ref:
            zip_ref.extractall(extraction_folder)
        
        # Remove the zip file
        os.remove(zip_name)
        
        return f'[+] {dir_name} Model successfully downloaded and extracted!'
    except Exception as e:
        raise Exception(f"Error downloading model: {str(e)}")

def get_rvc_model(voice_model):
    model_dir = os.path.join(rvc_models_dir, voice_model)
    for file in os.listdir(model_dir):
        if file.endswith('.pth'):
            return os.path.join(model_dir, file)
    raise FileNotFoundError(f"No .pth file found in RVC model directory: {model_dir}")

def voice_conversion(input_audio, rvc_model, pitch=0, f0_method='rmvpe', index_rate=0.5, filter_radius=3, rms_mix_rate=0.25, protect=0.33):
    try:
        hubert_model = load_hubert(device, is_half, os.path.join(rvc_models_dir, "hubert_base.pt"))
        model_path = get_rvc_model(rvc_model)
        cpt, version, net_g, tgt_sr, vc = get_vc(device, is_half, Config(device, is_half), model_path)

        output_filename = os.path.join(output_dir, f"converted_{os.path.basename(input_audio)}")
        output_filename = os.path.splitext(output_filename)[0] + '.wav'
        os.makedirs(output_dir, exist_ok=True)

        rvc_infer("", index_rate, input_audio, output_filename, pitch, f0_method, cpt, version, net_g, 
                  filter_radius, tgt_sr, rms_mix_rate, protect, 160, vc, hubert_model)

        return output_filename
    except Exception as e:
        raise Exception(f"Voice conversion failed: {str(e)}")

def print_example_usage():
    print("\nUsage:")
    print('python main.py <input_audio> <rvc_model> [pitch] [f0_method] [index_rate] [filter_radius] [rms_mix_rate] [protect]')
    print("\nRequired arguments:")
    print("  input_audio: path to input audio file")
    print("  rvc_model: name of the RVC model to use")
    print("\nOptional arguments:")
    print("  pitch: pitch shift (default: 0)")
    print("  f0_method: pitch extraction method (default: 'rmvpe')")
    print("  index_rate: index rate (default: 0.5)")
    print("  filter_radius: filter radius (default: 3)")
    print("  rms_mix_rate: RMS mix rate (default: 0.25)")
    print("  protect: protect rate (default: 0.33)")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Error: Insufficient arguments.")
        print_example_usage()
        sys.exit(1)

    try:
        input_audio = sys.argv[1]
        rvc_model = sys.argv[2]
        pitch = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        f0_method = sys.argv[4] if len(sys.argv) > 4 else 'rmvpe'
        index_rate = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
        filter_radius = int(sys.argv[6]) if len(sys.argv) > 6 else 3
        rms_mix_rate = float(sys.argv[7]) if len(sys.argv) > 7 else 0.25
        protect = float(sys.argv[8]) if len(sys.argv) > 8 else 0.33

        output_path = voice_conversion(input_audio, rvc_model, pitch, f0_method, index_rate, filter_radius, rms_mix_rate, protect)
        print(f"Converted audio saved to: {output_path}")
    except Exception as e:
        print(f"Error: {str(e)}")
        print_example_usage()
