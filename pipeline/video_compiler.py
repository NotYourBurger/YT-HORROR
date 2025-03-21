import os
import subprocess
import tempfile
import json
from typing import Dict, List, Tuple
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

class VideoCompiler:
    """Compiles all generated content into a final cinematic video"""
    
    def __init__(self, video_quality="4000k", cinematic_ratio=2.35, use_dust_overlay=True):
        """Initialize with video preferences"""
        self.video_quality = video_quality
        self.cinematic_ratio = float(cinematic_ratio)
        self.use_dust_overlay = use_dust_overlay
        
        # For dust and scratch overlay
        self.dust_overlay_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'assets', 'dust_overlay.png'
        )
        
        # Create assets directory if it doesn't exist
        os.makedirs(os.path.dirname(self.dust_overlay_path), exist_ok=True)
        
        # Create a simple dust overlay if it doesn't exist
        if not os.path.exists(self.dust_overlay_path):
            self._create_placeholder_dust_overlay()
    
    def _create_placeholder_dust_overlay(self):
        """Create a simple dust overlay texture"""
        width, height = 1920, 1080
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Add random dust particles
        for _ in range(500):
            x = random.randint(0, width)
            y = random.randint(0, height)
            size = random.randint(1, 3)
            opacity = random.randint(30, 100)
            draw.ellipse((x, y, x+size, y+size), fill=(255, 255, 255, opacity))
        
        # Add random scratches
        for _ in range(20):
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            x2 = x1 + random.randint(-200, 200)
            y2 = y1 + random.randint(-200, 200)
            width = random.randint(1, 2)
            opacity = random.randint(30, 70)
            draw.line((x1, y1, x2, y2), fill=(255, 255, 255, opacity), width=width)
        
        # Apply blur to make it more natural
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Save the overlay
        overlay.save(self.dust_overlay_path)
        print(f"Created placeholder dust overlay: {self.dust_overlay_path}")
    
    def _create_animation_script(self, image_paths: List[str], durations: List[Tuple[float, float]]) -> Dict:
        """Create animation script for each image in the sequence"""
        animations = []
        
        for i, (image_path, (start_time, end_time)) in enumerate(zip(image_paths, durations)):
            duration = end_time - start_time
            
            # Choose a random animation type
            animation_type = random.choice([
                'zoom_in', 'zoom_out', 'pan_left', 'pan_right', 'pan_up', 'pan_down'
            ])
            
            # Create animation parameters
            if animation_type == 'zoom_in':
                start_scale = 1.0
                end_scale = 1.0 + (0.1 * min(duration, 10))
                animation = {
                    'type': 'zoom',
                    'start_scale': start_scale,
                    'end_scale': end_scale,
                    'center_x': 0.5,
                    'center_y': 0.5
                }
            elif animation_type == 'zoom_out':
                start_scale = 1.0 + (0.1 * min(duration, 10))
                end_scale = 1.0
                animation = {
                    'type': 'zoom',
                    'start_scale': start_scale,
                    'end_scale': end_scale,
                    'center_x': 0.5,
                    'center_y': 0.5
                }
            elif animation_type.startswith('pan'):
                # Determine pan direction
                if animation_type == 'pan_left':
                    start_x, start_y = 0.45, 0.5
                    end_x, end_y = 0.55, 0.5
                elif animation_type == 'pan_right':
                    start_x, start_y = 0.55, 0.5
                    end_x, end_y = 0.45, 0.5
                elif animation_type == 'pan_up':
                    start_x, start_y = 0.5, 0.45
                    end_x, end_y = 0.5, 0.55
                else:  # pan_down
                    start_x, start_y = 0.5, 0.55
                    end_x, end_y = 0.5, 0.45
                
                animation = {
                    'type': 'pan',
                    'start_x': start_x,
                    'start_y': start_y,
                    'end_x': end_x,
                    'end_y': end_y,
                    'scale': 1.05  # Slight zoom to avoid blank edges
                }
            
            animations.append({
                'image_path': image_path,
                'start_time': start_time,
                'duration': duration,
                'animation': animation
            })
        
        return {'animations': animations}
    
    def _apply_cinematic_ratio(self, image_path: str, output_path: str) -> str:
        """Apply cinematic black bars based on desired aspect ratio"""
        try:
            # Open the image
            image = Image.open(image_path)
            width, height = image.size
            
            # Calculate the height for the target aspect ratio
            target_height = int(width / self.cinematic_ratio)
            
            if target_height < height:
                # Create a new black image
                cinematic_image = Image.new('RGB', (width, height), (0, 0, 0))
                
                # Calculate the position to center the original image
                y_offset = (height - target_height) // 2
                
                # Crop the original image to the target aspect ratio
                cropped = image.crop((0, (height - target_height) // 2, width, (height + target_height) // 2))
                
                # Paste the cropped image onto the black background
                cinematic_image.paste(cropped, (0, y_offset))
            else:
                # No need to add black bars if the image is already wider than the target ratio
                cinematic_image = image
            
            # Save the result
            cinematic_image.save(output_path)
            
            return output_path
            
        except Exception as e:
            print(f"Error applying cinematic ratio to {image_path}: {str(e)}")
            # If it fails, just copy the original image
            import shutil
            shutil.copy(image_path, output_path)
            return output_path
    
    def _add_dust_overlay(self, image_path: str, output_path: str) -> str:
        """Add dust and scratch overlay for vintage film effect"""
        try:
            # Check if dust overlay exists
            if not os.path.exists(self.dust_overlay_path):
                print("Dust overlay not found, skipping effect")
                import shutil
                shutil.copy(image_path, output_path)
                return output_path
                
            # Open the image and dust overlay
            image = Image.open(image_path)
            dust = Image.open(self.dust_overlay_path)
            
            # Resize dust overlay to match image size
            dust = dust.resize(image.size)
            
            # Composite the images
            if dust.mode == 'RGBA':
                # If the dust overlay has alpha channel
                result = Image.alpha_composite(image.convert('RGBA'), dust)
                result = result.convert('RGB')
            else:
                # If it's a regular RGB image, use blend mode
                opacity = 0.15  # Subtle effect
                result = Image.blend(image, dust.convert('RGB'), opacity)
            
            # Save the result
            result.save(output_path)
            
            return output_path
            
        except Exception as e:
            print(f"Error adding dust overlay to {image_path}: {str(e)}")
            # If it fails, just copy the original image
            import shutil
            shutil.copy(image_path, output_path)
            return output_path
    
    def create_final_video(
        self, 
        image_prompts: List[Dict], 
        image_paths: List[str],
        audio_path: str, 
        srt_path: str,
        ambient_path: str,
        title: str
    ) -> str:
        """Create the final video with images, audio, subtitles, and effects"""
        try:
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"Creating final video for '{title}'")
                
                # Prepare timing information
                durations = [(
                    self._convert_timestamp_to_seconds(prompt['timing'][0]), 
                    self._convert_timestamp_to_seconds(prompt['timing'][1])
                ) for prompt in image_prompts]
                
                # Create processed images directory
                processed_dir = os.path.join(temp_dir, 'processed_images')
                os.makedirs(processed_dir, exist_ok=True)
                
                # Process each image with effects
                processed_image_paths = []
                for i, image_path in enumerate(image_paths):
                    # Apply cinematic ratio
                    ratio_path = os.path.join(processed_dir, f"ratio_{i}.jpg")
                    self._apply_cinematic_ratio(image_path, ratio_path)
                    
                    # Apply dust overlay if requested
                    if self.use_dust_overlay:
                        final_path = os.path.join(processed_dir, f"final_{i}.jpg")
                        self._add_dust_overlay(ratio_path, final_path)
                    else:
                        final_path = ratio_path
                    
                    processed_image_paths.append(final_path)
                
                # Create animation script
                animation_script = self._create_animation_script(processed_image_paths, durations)
                animation_script_path = os.path.join(temp_dir, 'animation.json')
                with open(animation_script_path, 'w') as f:
                    json.dump(animation_script, f, indent=2)
                
                # Create output path
                output_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'storage')
                os.makedirs(output_folder, exist_ok=True)
                output_path = os.path.join(output_folder, f"{title.replace(' ', '_')}.mp4")
                
                # For a real implementation, we would use FFmpeg to create the video
                # For this example, we'll simulate the process
                print("Simulating video creation with FFmpeg...")
                print(f"- Using {len(processed_image_paths)} processed images")
                print(f"- Audio: {audio_path}")
                print(f"- Ambient: {ambient_path}")
                print(f"- Subtitles: {srt_path}")
                print(f"- Video quality: {self.video_quality}")
                
                # Create a simple text file to show the command that would be executed
                command_path = os.path.join(output_folder, f"{title.replace(' ', '_')}_command.txt")
                with open(command_path, 'w') as f:
                    f.write(f"""
FFmpeg command that would be executed to create the video:

ffmpeg -y \\
  -f concat -safe 0 -i scenes.txt \\
  -i "{audio_path}" \\
  -i "{ambient_path}" \\
  -c:v libx264 -preset slow -crf 22 -pix_fmt yuv420p \\
  -c:a aac -b:a 192k \\
  -filter_complex "[1:a][2:a]amix=inputs=2:duration=longest:dropout_transition=3,volume=2" \\
  -vf "subtitles={srt_path}" \\
  -b:v {self.video_quality} \\
  -movflags +faststart \\
  "{output_path}"
                    """)
                
                # In a real implementation, we would execute ffmpeg here
                # For this example, we'll just copy the first image as the "video"
                import shutil
                if processed_image_paths:
                    shutil.copy(processed_image_paths[0], output_path + ".jpg")
                    print(f"Created placeholder video preview: {output_path}.jpg")
                
                print(f"Video command file created: {command_path}")
                print(f"In a real implementation, the video would be saved to: {output_path}")
                
                return output_path
                
        except Exception as e:
            print(f"Error creating final video: {str(e)}")
            raise
    
    def _convert_timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert SRT timestamp format to seconds"""
        parts = timestamp.replace(',', '.').split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds 