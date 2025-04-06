import folder_paths
import re
from pathlib import Path
import requests
import os

CIVITAI_URL = "https://civitai.com/api/download/models"

# Support both CIVITAI_API_KEY and civitai_token as ENV vars
CIVITAI_API_KEY = os.environ.get("CIVITAI_API_KEY") or os.environ.get("civitai_token")

class WanVideoLoraTagLoader:
    def __init__(self):
        self.tag_pattern = r"<lora:([\w\-\.]+)(?::([\d\.]+))?>"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            },
            "optional": {
                "prev_lora":("WANVIDLORA", {"default": None, "tooltip": "For loading multiple LoRAs"}),
                "blocks":("SELECTEDBLOCKS", ),
                "low_mem_load": ("BOOLEAN", {"default": False, "tooltip": "Load with less VRAM usage"}),
            }
        }

    RETURN_TYPES = ("WANVIDLORA", "STRING")
    RETURN_NAMES = ("lora", "text")
    FUNCTION = "parse_and_load"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Parses <lora:name:strength> tags in the prompt and loads WAN-compatible LoRAs. Will attempt to download from Civitai if not found."

    def parse_and_load(self, text, prev_lora=None, blocks=None, low_mem_load=False):
        founds = re.findall(self.tag_pattern, text)
        loras_list = []

        if prev_lora is not None:
            loras_list.extend(prev_lora)

        lora_files = folder_paths.get_filename_list("loras")
        lora_dir = folder_paths.get_folder_paths("loras")[0]

        for name, strength in founds:
            lora_name = next((f for f in lora_files if f.startswith(name)), None)
            if lora_name is None:
                print(f"[WanVideoLoraTagLoader] Trying to download {name} from Civitai...")
                lora_name = self.download_lora_from_civitai(name, lora_dir)
                if lora_name is None:
                    print(f"[WanVideoLoraTagLoader] Skipping unknown LoRA: {name}")
                    continue

            try:
                weight = float(strength) if strength else 1.0
            except ValueError:
                weight = 1.0

            lora_entry = {
                "path": folder_paths.get_full_path("loras", lora_name),
                "strength": weight,
                "name": lora_name.split(".")[0],
                "blocks": blocks,
                "low_mem_load": low_mem_load,
            }
            print(f"[WanVideoLoraTagLoader] Using LoRA: {lora_entry}")
            loras_list.append(lora_entry)

        cleaned_text = re.sub(self.tag_pattern, "", text).strip()

        return (loras_list, cleaned_text)

    def download_lora_from_civitai(self, model_name, lora_dir):
        try:
            # Search Civitai for the model
            search_url = f"https://civitai.com/api/v1/models?query={model_name}&types=LORA"
            headers = {"Authorization": f"Bearer {CIVITAI_API_KEY}"} if CIVITAI_API_KEY else {}
            r = requests.get(search_url, headers=headers)
            if r.status_code != 200:
                print(f"[WanVideoLoraTagLoader] Failed search: {r.status_code} {r.text}")
                return None
            data = r.json()
            if not data['items']:
                return None

            # Grab latest version and file
            model_id = data['items'][0]['modelVersions'][0]['id']
            files = data['items'][0]['modelVersions'][0]['files']
            safetensor = next((f for f in files if f['name'].endswith(".safetensors")), None)
            if safetensor is None:
                return None

            file_url = safetensor['downloadUrl']
            out_path = Path(lora_dir) / safetensor['name']
            with requests.get(file_url, stream=True, headers=headers) as resp:
                resp.raise_for_status()
                with open(out_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)

            print(f"[WanVideoLoraTagLoader] Downloaded {safetensor['name']} to {out_path}")
            return safetensor['name']

        except Exception as e:
            print(f"[WanVideoLoraTagLoader] Failed to download {model_name}: {e}")
            return None


NODE_CLASS_MAPPINGS = {
    "WanVideoLoraTagLoader": WanVideoLoraTagLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoLoraTagLoader": "WAN Load LoRA Tag",
}
