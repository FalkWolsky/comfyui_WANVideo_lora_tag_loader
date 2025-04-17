WanVideo LoRA Tag Loader, Dynamic Text Encoder, and LoRA Search for ComfyUI
=======

A ComfyUI custom node extension that provides three powerful nodes for working with WAN Video:
1. **WanVideo Auto-Load LoRA Tags** - Automatically loads LoRAs from text prompts
2. **WanVideo Dynamic Text Encoder** - Advanced text encoding with prompt weighting and memory management
3. **WanVideo LoRA Search** - Advanced LoRA search functionality with metadata support

## Features

### WanVideo Auto-Load LoRA Tags
- Automatically extracts and loads LoRA tags from text prompts
- Supports multiple LoRAs in a single prompt
- Automatic Civitai integration for downloading missing LoRAs
- Format: `<lora:name:strength>`
- Memory-efficient loading option
- Automatic metadata collection and storage
- Support for LoRA blocks and low memory loading

### WanVideo Dynamic Text Encoder
- Advanced text encoding with prompt weighting support
- Memory management with automatic model offloading
- Support for prompt overrides
- Weighted prompt syntax: `(text:weight)`
- Multiple positive prompts support with `|` separator
- Automatic VRAM optimization
- Support for model offloading during encoding
- Flexible device management

### WanVideo LoRA Search
- Search LoRAs by tags and trained words
- Filter by NSFW level and thumbs up count
- Sort by popularity, NSFW level, or name
- Availability filtering (Public/Private)
- REST API endpoint for programmatic access
- Comprehensive metadata search
- Grouped results with tags and trained words
- Flexible sorting options (ascending/descending)

## Installation
1. Clone or download this repository
2. Place the `comfyui_WANVideo_lora_tag_loader` folder into your ComfyUI's `custom_nodes` directory
3. Restart ComfyUI

## Usage Examples

### LoRA Tag Loader
```text
# Basic usage
<lora:model_name:0.8>

# Multiple LoRAs
<lora:first_model:0.8> <lora:second_model:0.5>

# Memory efficient loading
<lora:model_name:0.8> (with low_mem_load enabled)

# With blocks specification
<lora:model_name:0.8> (with specific blocks)
```

### Dynamic Text Encoder
```text
# Basic weighted prompts
(beautiful:1.2) (detailed:1.3) landscape

# Multiple positive prompts
(beautiful:1.2) landscape | (detailed:1.3) scenery

# With negative prompt
Negative: blurry, low quality

# With model offloading
(force_offload enabled)
```

### LoRA Search
```text
# Search by tags
Search Type: TAGS
Search Terms: anime, style

# Search by trained words
Search Type: TRAINED_WORDS
Search Terms: portrait, style

# Advanced search with filters
Min Thumbs Up: 100
Max NSFW Level: 50
Availability: Public
Sort By: THUMBS_UP
Sort Order: DESC
```

## API Usage
The LoRA Search functionality is available via a REST API endpoint:
```http
POST /wanvideo/lora/search
Content-Type: application/json

{
    "search_type": "ALL",
    "search_terms": "anime, style",
    "min_thumbs_up": 100,
    "max_nsfw_level": 50,
    "availability": "Public",
    "sort_by": "THUMBS_UP",
    "sort_order": "DESC"
}
```

## Requirements
- ComfyUI
- WAN Video extension
- Civitai API key (optional, for automatic LoRA downloads)
- SQLite3 (for metadata storage)

## Configuration
Set your Civitai API key as an environment variable:
```bash
export CIVITAI_API_KEY="your_api_key_here"
```
or
```bash
export civitai_token="your_api_key_here"
```

## Metadata Database
The extension automatically maintains a SQLite database (`lora_metadata.db`) that stores:
- LoRA metadata
- Tags
- Trained words
- NSFW levels
- Thumbs up counts
- Availability status

## License
This project is licensed under the terms included in the LICENSE file.
