WanVideo LoRA Tag Loader and Dynamic Text Encoder for ComfyUI
=======

A ComfyUI custom node extension that provides two powerful nodes for working with WAN Video:
1. **WanVideo Auto-Load LoRA Tags** - Automatically loads LoRAs from text prompts
2. **WanVideo Dynamic Text Encoder** - Advanced text encoding with prompt weighting and memory management

## Features

### WanVideo Auto-Load LoRA Tags
- Automatically extracts and loads LoRA tags from text prompts
- Supports multiple LoRAs in a single prompt
- Automatic Civitai integration for downloading missing LoRAs
- Format: `<lora:name:strength>`

### WanVideo Dynamic Text Encoder
- Advanced text encoding with prompt weighting support
- Memory management with automatic model offloading
- Support for prompt overrides
- Weighted prompt syntax: `(text:weight)`
- Multiple positive prompts support with `|` separator
- Automatic VRAM optimization

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
```

### Dynamic Text Encoder
```text
# Basic weighted prompts
(beautiful:1.2) (detailed:1.3) landscape

# Multiple positive prompts
(beautiful:1.2) landscape | (detailed:1.3) scenery

# With negative prompt
Negative: blurry, low quality
```

## Requirements
- ComfyUI
- WAN Video extension
- Civitai API key (optional, for automatic LoRA downloads)

## Configuration
Set your Civitai API key as an environment variable:
```bash
export CIVITAI_API_KEY="your_api_key_here"
```
or
```bash
export civitai_token="your_api_key_here"
```

## License
This project is licensed under the terms included in the LICENSE file.
