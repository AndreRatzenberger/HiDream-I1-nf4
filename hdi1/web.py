import torch
import gradio as gr
import logging
import os
import argparse

from .nf4 import *

# Resolution options
RESOLUTION_OPTIONS = [
    "1024 × 1024 (Square)",
    "768 × 1360 (Portrait)",
    "1360 × 768 (Landscape)",
    "880 × 1168 (Portrait)",
    "1168 × 880 (Landscape)",
    "1248 × 832 (Landscape)",
    "832 × 1248 (Portrait)"
]

# Parse resolution string to get height and width
def parse_resolution(resolution_str):
    return tuple(map(int, resolution_str.split("(")[0].strip().split(" × ")))


def gen_img_helper(model, custom_path, prompt, res, seed):
    global pipe, current_model, current_custom_path
    
    # Determine if we're using a predefined or custom model
    use_custom = custom_path.strip() != ""
    
    # Get the model path and identifier
    model_identifier = "custom" if use_custom else model
    model_path = custom_path if use_custom else model
    
    # Check if model needs to be reloaded
    need_reload = model_identifier != current_model or (use_custom and custom_path != current_custom_path)
    
    # 1. Check if the model matches loaded model, load the model if not
    if need_reload:
        print(f"Unloading model {current_model}...")
        del pipe
        torch.cuda.empty_cache()
        
        if use_custom:
            # Validate custom path
            if not os.path.exists(custom_path):
                return None, f"Error: Custom model path '{custom_path}' does not exist"
            
            print(f"Loading custom model from {custom_path}...")
            pipe, _ = load_custom_model(custom_path)
            current_model = "custom"
            current_custom_path = custom_path
        else:
            print(f"Loading model {model}...")
            pipe, _ = load_models(model)
            current_model = model
            current_custom_path = ""
            
        print("Model loaded successfully!")

    # 2. Generate image
    res = parse_resolution(res)
    try:
        image, used_seed = generate_image(pipe, model_identifier, prompt, res, seed)
        return image, used_seed
    except Exception as e:
        return None, f"Error generating image: {str(e)}"


if __name__ == "__main__":
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    
    # Setup command-line argument parser
    parser = argparse.ArgumentParser(description="HiDream-I1-nf4 Web Interface")
    parser.add_argument("-m", "--model", type=str, default="fast",
                       help="Model to use at startup",
                       choices=["dev", "full", "fast"])
    parser.add_argument("-p", "--path", type=str, default="",
                       help="Path to a custom model directory to use at startup")
    
    # Add gradio server args
    parser.add_argument("--share", action="store_true", help="Create a public URL")
    parser.add_argument("--port", type=int, help="Port to run the server on")
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="Server address")
    
    args = parser.parse_args()
    
    # Initialize with specified model
    current_custom_path = ""
    
    if args.path:
        # User provided a custom model path
        if not os.path.exists(args.path):
            print(f"Error: Custom model path '{args.path}' does not exist")
            exit(1)
        
        print(f"Loading custom model from {args.path}...")
        current_model = "custom"
        current_custom_path = args.path
        pipe, _ = load_custom_model(args.path)
    else:
        # User specified a predefined model or using default
        print(f"Loading model {args.model}...")
        current_model = args.model
        pipe, _ = load_models(args.model)
    
    print("Model loaded successfully!")

    # Determine initial UI values based on startup configuration
    initial_model = args.model if not args.path else "fast"  # Default to fast if custom path is used
    initial_path = args.path

    # Create Gradio interface
    with gr.Blocks(title="HiDream-I1-nf4 Dashboard") as demo:
        gr.Markdown("# HiDream-I1-nf4 Dashboard")
        
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    gr.Markdown("## Model Selection")
                    model_type = gr.Radio(
                        choices=list(k for k in MODEL_CONFIGS.keys() if k != "custom"),
                        value=initial_model,
                        label="Predefined Model",
                        info="Select a predefined model variant"
                    )
                    
                    custom_model_path = gr.Textbox(
                        label="Custom Model Path (Optional)", 
                        placeholder="/path/to/your/model",
                        info="If provided, this will override the predefined model selection",
                        value=initial_path
                    )
                
                prompt = gr.Textbox(
                    label="Prompt", 
                    placeholder="A cat holding a sign that says \"Hi-Dreams.ai\".", 
                    lines=3
                )
                
                resolution = gr.Radio(
                    choices=RESOLUTION_OPTIONS,
                    value=RESOLUTION_OPTIONS[0],
                    label="Resolution",
                    info="Select image resolution"
                )
                
                seed = gr.Number(
                    label="Seed (use -1 for random)", 
                    value=-1, 
                    precision=0
                )
                
                generate_btn = gr.Button("Generate Image")
                seed_used = gr.Number(label="Seed Used", interactive=False)
                
            with gr.Column():
                output_image = gr.Image(label="Generated Image", type="pil")
        
        generate_btn.click(
            fn=gen_img_helper,
            inputs=[model_type, custom_model_path, prompt, resolution, seed],
            outputs=[output_image, seed_used]
        )

    # Launch with the specified server options
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share
    )
