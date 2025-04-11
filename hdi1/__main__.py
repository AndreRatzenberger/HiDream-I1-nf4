from .nf4 import *

import argparse
import time
import logging
import os


if __name__ == "__main__":
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("prompt", type=str, help="Prompt to generate image from")
    
    parser.add_argument("-m", "--model", type=str, default="dev",
                        help="Model to use",
                        choices=["dev", "full", "fast"])
    
    parser.add_argument("-p", "--path", type=str, default=None,
                        help="Path to a custom model directory")
    
    parser.add_argument("-s", "--seed", type=int, default=-1, 
                        help="Seed for generation")
    
    parser.add_argument("-r", "--res", type=str, default="1024x1024", 
                        help="Resolution for generation", 
                        choices=["1024x1024", "768x1360", "1360x768", "880x1168", "1168x880", "1248x832", "832x1248"])
    
    parser.add_argument("-o", "--output", type=str, default="output.png")
    
    args = parser.parse_args()
    
    # Initialize model
    if args.path:
        if not os.path.exists(args.path):
            print(f"Error: Custom model path '{args.path}' does not exist")
            exit(1)
        print(f"Loading custom model from {args.path}...")
        pipe, config = load_custom_model(args.path)
        model_type = "custom"
    else:
        model_type = args.model
        print(f"Loading predefined model {model_type}...")
        pipe, config = load_models(model_type)
    
    print("Model loaded successfully!")
    
    st = time.time()
    
    resolution = tuple(map(int, args.res.strip().split("x")))
    image, seed = generate_image(pipe, model_type, args.prompt, resolution, args.seed)
    image.save(args.output)
    
    print(f"Image saved to {args.output}, elapsed time: {time.time() - st:.2f} seconds")
    print(f"Seed used: {seed}")
