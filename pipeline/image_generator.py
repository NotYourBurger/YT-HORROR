import os
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import random
from typing import Dict, Optional

class ImageGenerator:
    """Generates cinematic images from scene descriptions"""
    
    def __init__(self, style="cinematic"):
        """Initialize with style preferences"""
        self.style = style
        self.style_guidance = {
            "realistic": "photorealistic, intricate details, natural lighting, cinematic photography, 8k resolution, dramatic composition",
            "cinematic": "cinematic composition, dramatic lighting, film grain, anamorphic lens effect, professional cinematography, color grading, depth of field",
            "artistic": "digital art, stylized, vibrant colors, dramatic composition, concept art, trending on artstation, by Greg Rutkowski and Zdzisław Beksiński",
            "neutral": "balanced composition, masterful photography, perfect exposure, selective focus, attention-grabbing depth of field, highly atmospheric"
        }
        
        # Check if Stable Diffusion model is installed
        self.model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'models', 'stable_diffusion'
        )
        
        if not os.path.exists(self.model_path):
            print("Warning: Stable Diffusion model directory not found. Image generation may fail.")
        
        self.sd_pipeline = None
    
    def _initialize_pipeline(self):
        """Initialize Stable Diffusion pipeline"""
        try:
            # Load SDXL model
            self.sd_pipeline = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            ).to("cuda")
            
            # Set optimal scheduler
            self.sd_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.sd_pipeline.scheduler.config,
                algorithm_type="sde-dpmsolver++",
                use_karras_sigmas=True
            )
            
            print("Stable Diffusion XL pipeline initialized successfully")
            
        except Exception as e:
            print(f"Error initializing Stable Diffusion: {str(e)}")
            raise
    
    def enhance_prompt(self, prompt: str) -> str:
        """Add standard enhancement terms to a prompt"""
        return f"{prompt}, highly detailed, cinematic lighting, atmospheric, 8k resolution"
    
    def generate_image(self, description: str, output_path: str) -> str:
        """Generate image from scene description"""
        try:
            # Initialize pipeline if needed
            if self.sd_pipeline is None:
                self._initialize_pipeline()
            
            # Create a cinematic prompt with optimal structure
            style_desc = self.style_guidance.get(self.style, self.style_guidance["cinematic"])
            cinematic_prompt = f"{description}, {style_desc}"
            
            # Refine the prompt
            enhanced_prompt = self.enhance_prompt(cinematic_prompt)
            
            # Create a refined negative prompt
            negative_prompt = (
                "low quality, blurry, distorted, deformed, disfigured, bad anatomy, "
                "bad proportions, extra limbs, missing limbs, disconnected limbs, "
                "duplicate, mutated, ugly, watermark, watermarked, text, signature, "
                "logo, oversaturated, cartoon, 3d render, bad art, amateur, "
                "poorly drawn face, poorly drawn hands, poorly drawn feet"
            )
            
            # Generate a random seed for variety but allow reproducibility
            seed = random.randint(1, 2147483647)
            torch_generator = torch.Generator(device="cuda").manual_seed(seed)
            
            # Aspect ratios optimized for SDXL (using 3:2 for cinematic look)
            width, height = 1024, 680  # 3:2 aspect ratio, optimized for SDXL
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Generate the image
            image = self.sd_pipeline(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=40,     # Higher step count for better quality
                guidance_scale=7.5,         # Optimal CFG value for SDXL
                generator=torch_generator,
                output_type="pil"
            ).images[0]
            
            # Save the image
            image.save(output_path)
            
            print(f"Image generated: {output_path} (seed: {seed})")
            return output_path
            
        except Exception as e:
            print(f"Error generating image: {str(e)}")
            raise 