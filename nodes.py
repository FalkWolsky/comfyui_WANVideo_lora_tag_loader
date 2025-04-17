import folder_paths
import re
from pathlib import Path
import requests
import os
import sqlite3
from typing import List, Dict, Any
import json

CIVITAI_URL = "https://civitai.com/api/download/models"

# Support both CIVITAI_API_KEY and civitai_token as ENV vars
CIVITAI_API_KEY = os.environ.get("CIVITAI_API_KEY") or os.environ.get("civitai_token")

class LoraMetadataDB:
    def __init__(self, db_path: str = "lora_metadata.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            # Create main lora table
            c.execute('''CREATE TABLE IF NOT EXISTS loras
                        (model_name_safe TEXT PRIMARY KEY,
                         nsfw_level INTEGER,
                         availability TEXT,
                         thumbs_up_count INTEGER,
                         model_id INTEGER)''')
            
            # Create tags table with many-to-many relationship
            c.execute('''CREATE TABLE IF NOT EXISTS tags
                        (id INTEGER PRIMARY KEY,
                         tag TEXT UNIQUE)''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS lora_tags
                        (lora_name TEXT,
                         tag_id INTEGER,
                         FOREIGN KEY(lora_name) REFERENCES loras(model_name_safe),
                         FOREIGN KEY(tag_id) REFERENCES tags(id),
                         PRIMARY KEY(lora_name, tag_id))''')
            
            # Create trained words table with many-to-many relationship
            c.execute('''CREATE TABLE IF NOT EXISTS trained_words
                        (id INTEGER PRIMARY KEY,
                         word TEXT UNIQUE)''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS lora_trained_words
                        (lora_name TEXT,
                         word_id INTEGER,
                         FOREIGN KEY(lora_name) REFERENCES loras(model_name_safe),
                         FOREIGN KEY(word_id) REFERENCES trained_words(id),
                         PRIMARY KEY(lora_name, word_id))''')
            
            # Create indexes for better search performance
            c.execute('CREATE INDEX IF NOT EXISTS idx_nsfw ON loras(nsfw_level)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_thumbs ON loras(thumbs_up_count)')
            conn.commit()

    def add_lora(self, model_name_safe: str, metadata: Dict[str, Any]):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            
            # Insert main lora data
            c.execute('''INSERT OR REPLACE INTO loras 
                        (model_name_safe, nsfw_level, availability, thumbs_up_count, model_id)
                        VALUES (?, ?, ?, ?, ?)''',
                     (model_name_safe,
                      metadata.get('preview_nsfw_level', 0),
                      metadata.get('civitai', {}).get('availability', 'Public'),
                      metadata.get('civitai', {}).get('stats', {}).get('thumbsUpCount', 0),
                      metadata.get('civitai', {}).get('id', 0)))
            
            # Add tags
            for tag in metadata.get('tags', []):
                c.execute('INSERT OR IGNORE INTO tags (tag) VALUES (?)', (tag,))
                c.execute('SELECT id FROM tags WHERE tag = ?', (tag,))
                tag_id = c.fetchone()[0]
                c.execute('INSERT OR IGNORE INTO lora_tags (lora_name, tag_id) VALUES (?, ?)',
                         (model_name_safe, tag_id))
            
            # Add trained words
            for word in metadata.get('usage_tips', {}).get('trainedWords', []):
                c.execute('INSERT OR IGNORE INTO trained_words (word) VALUES (?)', (word,))
                c.execute('SELECT id FROM trained_words WHERE word = ?', (word,))
                word_id = c.fetchone()[0]
                c.execute('INSERT OR IGNORE INTO lora_trained_words (lora_name, word_id) VALUES (?, ?)',
                         (model_name_safe, word_id))
            
            conn.commit()

class WanVideoLoraTagLoader:
    def __init__(self):
        # Allow spaces and special chars like ':' in LoRA names
        self.tag_pattern = r"<lora:([^:>]+)(?::([\d\.]+))?>"
        self.db = LoraMetadataDB()

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
        print(f"[WanVideoLoraTagLoader] Current founds: {founds}")

        loras_list = []

        if prev_lora is not None:
            loras_list.extend(prev_lora)

        lora_files = folder_paths.get_filename_list("loras")
        lora_files_lower = [f.lower() for f in lora_files]
        lora_dir = folder_paths.get_folder_paths("loras")[0]
        print(f"[WanVideoLoraTagLoader] Current File List: {lora_dir}, {lora_files_lower}")

        for name, strength in founds:
            name_lower = name.lower().strip()
            lora_index = next((i for i, f in enumerate(lora_files_lower) if name_lower in f), None)
            lora_name = lora_files[lora_index] if lora_index is not None else None

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
                "name": name.strip(),
                "blocks": blocks,
                "low_mem_load": low_mem_load,
            }
            print(f"[WanVideoLoraTagLoader] Using LoRA: {lora_entry}")
            loras_list.append(lora_entry)

        cleaned_text = re.sub(self.tag_pattern, "", text).strip()
        return (loras_list, cleaned_text)

    def download_lora_from_civitai(self, model_name, lora_dir):
        try:
            search_url = f"https://civitai.com/api/v1/models?query={model_name}&types=LORA"
            headers = {"Authorization": f"Bearer {CIVITAI_API_KEY}"} if CIVITAI_API_KEY else {}
            r = requests.get(search_url, headers=headers)
            if r.status_code != 200:
                print(f"[WanVideoLoraTagLoader] Failed search: {r.status_code} {r.text}")
                return None
            data = r.json()
            if not data['items']:
                return None

            model = data['items'][0]
            model_version = model['modelVersions'][0]
            model_id = model['id']
            model_name_safe = model['name'].lower()
            model_name_safe = re.sub(r'\s+', '-', model_name_safe)
            model_name_safe = re.sub(r'[^a-z0-9-]', '', model_name_safe)
            files = model_version['files']
            safetensor = next((f for f in files if f['name'].endswith(".safetensors")), None)
            if safetensor is None:
                return None

            # Create standard metadata structure
            standard_metadata = {
                "file_name": f"{model_id}_{model_name_safe}",
                "model_name": f"{model_id}_{model_name_safe}",
                "file_path": str(Path(lora_dir) / f"{model_id}_{model_name_safe}.safetensors"),
                "size": safetensor.get('sizeKB', 0) * 1024,  # Convert KB to bytes
                "modified": model_version.get('updatedAt', ''),
                "sha256": safetensor.get('hashes', {}).get('SHA256', ''),
                "base_model": model_version.get('baseModel', 'Unknown'),
                "preview_url": model_version.get('images', [{}])[0].get('url', '') if model_version.get('images') else '',
                "preview_nsfw_level": model.get('nsfwLevel', 0),
                "usage_tips": {
                    "description": model.get('description', ''),
                    "trainedWords": model_version.get('trainedWords', [])
                },
                "notes": "",
                "from_civitai": True,
                "civitai": model,  # Store the full civitai response
                "tags": model.get('tags', []),
                "modelDescription": model.get('description', '')
            }

            # Add metadata to database
            self.db.add_lora(model_name_safe, standard_metadata)

            # Save standard metadata file
            metadata_filename = f"{model_id}_{model_name_safe}.metadata.json"
            metadata_path = Path(lora_dir) / metadata_filename
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(standard_metadata, f, indent=2)
            print(f"[WanVideoLoraTagLoader] Saved metadata to {metadata_path}")

            # Use sanitized model name for filename
            filename = f"{model_id}_{model_name_safe}.safetensors"
            file_url = safetensor['downloadUrl']
            out_path = Path(lora_dir) / filename

            with requests.get(file_url, stream=True, headers=headers) as resp:
                resp.raise_for_status()
                with open(out_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)

            print(f"[WanVideoLoraTagLoader] Downloaded {filename} to {out_path}")
            return filename

        except Exception as e:
            print(f"[WanVideoLoraTagLoader] Failed to download {model_name}: {e}")
            return None

class WanVideoDynamicTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "t5": ("WANTEXTENCODER",),
                "positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True})
            },
            "optional": {
                "override_positive": ("STRING", {"multiline": True, "default": ""}),
                "override_negative": ("STRING", {"multiline": True, "default": ""}),
                "force_offload": ("BOOLEAN", {"default": True}),
                "model_to_offload": ("WANVIDEOMODEL", {"tooltip": "Model to move to offload_device before encoding"})
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", )
    RETURN_NAMES = ("text_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Encodes text prompts into text embeddings. Supports prompt overrides via input nodes."

    def process(self, t5, positive_prompt, negative_prompt, override_positive="", override_negative="", force_offload=True, model_to_offload=None):
        import torch
        import comfy.model_management as mm
        import comfy.utils as log

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        if model_to_offload is not None:
            log.info(f"Moving video model to {offload_device}")
            model_to_offload.model.to(offload_device)
            mm.soft_empty_cache()

        encoder = t5["model"]
        dtype = t5["dtype"]

        # Apply override if given
        if override_positive.strip():
            positive_prompt = override_positive
        if override_negative.strip():
            negative_prompt = override_negative

        # Split positive prompts and process each with weights
        positive_prompts_raw = [p.strip() for p in positive_prompt.split('|')]
        positive_prompts = []
        all_weights = []

        for p in positive_prompts_raw:
            cleaned_prompt, weights = self.parse_prompt_weights(p)
            positive_prompts.append(cleaned_prompt)
            all_weights.append(weights)

        encoder.model.to(device)

        with torch.autocast(device_type=mm.get_autocast_device(device), dtype=dtype, enabled=True):
            context = encoder(positive_prompts, device)
            context_null = encoder([negative_prompt], device)

            # Apply weights to embeddings if any were extracted
            for i, weights in enumerate(all_weights):
                for text, weight in weights.items():
                    log.info(f"Applying weight {weight} to prompt: {text}")
                    if len(weights) > 0:
                        context[i] = context[i] * weight

        if force_offload:
            encoder.model.to(offload_device)
            mm.soft_empty_cache()

        prompt_embeds_dict = {
            "prompt_embeds": context,
            "negative_prompt_embeds": context_null,
        }
        return (prompt_embeds_dict,)

    def parse_prompt_weights(self, prompt):
        """Extract text and weights from prompts with (text:weight) format"""
        import re

        # Parse all instances of (text:weight) in the prompt
        pattern = r'\((.*?):([\d\.]+)\)'
        matches = re.findall(pattern, prompt)

        # Replace each match with just the text part
        cleaned_prompt = prompt
        weights = {}

        for match in matches:
            text, weight = match
            orig_text = f"({text}:{weight})"
            cleaned_prompt = cleaned_prompt.replace(orig_text, text)
            weights[text] = float(weight)

        return cleaned_prompt, weights

class WanVideoLoraSearch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "search_type": (["TAGS", "TRAINED_WORDS", "ALL"], {"default": "ALL"}),
                "search_terms": ("STRING", {"multiline": True}),
                "min_thumbs_up": ("INT", {"default": 0, "min": 0}),
                "max_nsfw_level": ("INT", {"default": 100, "min": 0, "max": 100}),
                "availability": (["All", "Public", "Private"], {"default": "All"}),
                "sort_by": (["THUMBS_UP", "NSFW_LEVEL", "NAME"], {"default": "THUMBS_UP"}),
                "sort_order": (["DESC", "ASC"], {"default": "DESC"})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "search"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Search installed LoRAs by tags, trained words, and other criteria"

    def search(self, search_type, search_terms, min_thumbs_up, max_nsfw_level, 
              availability, sort_by, sort_order):
        db = LoraMetadataDB()
        terms = [t.strip() for t in search_terms.split(',') if t.strip()]
        
        with sqlite3.connect(db.db_path) as conn:
            c = conn.cursor()
            
            query = '''
                SELECT DISTINCT l.model_name_safe, l.nsfw_level, l.thumbs_up_count, l.availability
                FROM loras l
            '''
            
            params = []
            where_clauses = []
            
            if terms:
                if search_type in ['TAGS', 'ALL']:
                    query += '''
                        LEFT JOIN lora_tags lt ON l.model_name_safe = lt.lora_name
                        LEFT JOIN tags t ON lt.tag_id = t.id
                    '''
                if search_type in ['TRAINED_WORDS', 'ALL']:
                    query += '''
                        LEFT JOIN lora_trained_words lw ON l.model_name_safe = lw.lora_name
                        LEFT JOIN trained_words w ON lw.word_id = w.id
                    '''
                
                search_conditions = []
                for term in terms:
                    if search_type in ['TAGS', 'ALL']:
                        search_conditions.append('t.tag LIKE ?')
                        params.append(f'%{term}%')
                    if search_type in ['TRAINED_WORDS', 'ALL']:
                        search_conditions.append('w.word LIKE ?')
                        params.append(f'%{term}%')
                
                if search_conditions:
                    where_clauses.append('(' + ' OR '.join(search_conditions) + ')')
            
            if min_thumbs_up > 0:
                where_clauses.append('l.thumbs_up_count >= ?')
                params.append(min_thumbs_up)
            
            where_clauses.append('l.nsfw_level <= ?')
            params.append(max_nsfw_level)
            
            if availability != "All":
                where_clauses.append('l.availability = ?')
                params.append(availability)
            
            if where_clauses:
                query += ' WHERE ' + ' AND '.join(where_clauses)
            
            sort_column = {
                'THUMBS_UP': 'l.thumbs_up_count',
                'NSFW_LEVEL': 'l.nsfw_level',
                'NAME': 'l.model_name_safe'
            }[sort_by]
            
            query += f' ORDER BY {sort_column} {sort_order}'
            
            c.execute(query, params)
            results = c.fetchall()
            
            # Format results
            formatted_results = []
            for row in results:
                formatted_results.append(f"{row[0]} (👍 {row[2]}, NSFW: {row[1]}%, {row[3]})")
            
            return ("\n".join(formatted_results),)

NODE_CLASS_MAPPINGS = {
    "WanVideoLoraTagLoader": WanVideoLoraTagLoader,
    "WanVideoDynamicTextEncode": WanVideoDynamicTextEncode,
    "WanVideoLoraSearch": WanVideoLoraSearch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoLoraTagLoader": "WanVideo Auto-Load LoRA Tags",
    "WanVideoDynamicTextEncode": "WanVideo Dynamic Text Encoder",
    "WanVideoLoraSearch": "WanVideo LoRA Search"
}

