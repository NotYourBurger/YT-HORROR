# ===== CELL 1: AUTOMATED VOICE OVER GENERATION =====
# Import required libraries
import praw
from google import genai
import random
import whisper
import subprocess
import os
import re
import numpy as np
import soundfile as sf
from IPython.display import display, Audio
import torch
from diffusers import StableDiffusionPipeline
import time
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# ===== CELL 2: API CREDENTIALS =====
# Reddit API credentials
reddit = praw.Reddit(
    client_id="Jf3jkA3Y0dBCfluYvS8aVw",
    client_secret="1dWKIP6ME7FBR66motXS6273rkkf0g",
    user_agent="Horror Stories by Wear_Severe"
)

# Initialize Gemini API
client = genai.Client(api_key="AIzaSyD_vBSluRNPI6z7JoKfl67M6D3DCq4l0NI")


# ===== CELL 3: CONSTANTS =====
# Horror subreddits
HORROR_SUBREDDITS = ["nosleep", "shortscarystories", "creepypasta", "LetsNotMeet"]

# Horror themes for filtering and prompting
HORROR_THEMES = [
    "paranormal", "ghost", "haunting", "demon", "possession",
    "monster", "creature", "stalker", "serial killer", "unexplained",
    "ritual", "cult", "ancient evil", "cursed", "shadow people",
    "sleep paralysis", "night terror", "abandoned", "forest", "cabin",
    "basement", "attic", "mirror", "doppelganger", "entity"
]

# Global variable for Stable Diffusion model
sd_pipeline = None

# Constants for optimization
BG_MUSIC_DB = -17  # Background music level in dB


# ===== CELL 5: ENHANCED SCRIPT GENERATION (FETCH AND ENHANCE STORY) =====
def fetch_and_enhance_nosleep_story():
    """Fetch a story from horror subreddits and enhance it into a podcast format with intro/outro"""
    
    # Expanded list of horror subreddits
    horror_subreddits = [
        "nosleep", 
        "shortscarystories", 
        "creepypasta", 
        "LetsNotMeet",
        "DarkTales",
        "TheCrypticCompendium",
        "libraryofshadows",
        "scarystories",
        "TrueScaryStories",
        "HorrorStories"
    ]
    
    # Randomly select 2-3 subreddits to fetch from
    selected_subreddits = random.sample(horror_subreddits, min(3, len(horror_subreddits)))
    print(f"Fetching stories from: {', '.join(selected_subreddits)}")
    
    # Fetch stories from selected subreddits
    all_posts = []
    for subreddit_name in selected_subreddits:
        try:
            subreddit = reddit.subreddit(subreddit_name)
            # Get weekly top posts instead of monthly
            posts = list(subreddit.top("week", limit=30))
            all_posts.extend(posts)
            print(f"Found {len(posts)} stories from r/{subreddit_name}")
        except Exception as e:
            print(f"Error fetching from r/{subreddit_name}: {str(e)}")
    
    # Shuffle posts to randomize selection
    random.shuffle(all_posts)
    
    # Filter out very short posts and previously used posts
    cache_file = "used_story_ids.txt"
    used_ids = set()
    
    # Load previously used story IDs
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            used_ids = set(line.strip() for line in f.readlines())
    
    # Filter out previously used stories and very short ones
    filtered_posts = [
        post for post in all_posts 
        if post.id not in used_ids 
        and len(post.selftext) > 1000  # Ensure story has reasonable length
    ]
    
    if not filtered_posts:
        print("No new stories found, using all available stories")
        filtered_posts = [post for post in all_posts if len(post.selftext) > 1000]
    
    # Take a subset of posts for selection
    selection_posts = filtered_posts[:min(20, len(filtered_posts))]

    # Create a prompt for story selection
    post_titles = "\n".join([f"{i+1}. {post.title} (r/{post.subreddit.display_name})" for i, post in enumerate(selection_posts)])

    selection_prompt = """
    You are a professional horror story curator. Select ONE story from the following list
    that has the strongest potential for a cinematic narrative. Look for:
- Clear narrative structure
- Strong character development
    - Unique and original premise
    - Potential for visual storytelling
    - Atmospheric and sensory-rich content

Available stories:
    {titles}

    Return only the number of your selection (1-20).
    """.format(titles=post_titles)

    # Get story selection from Gemini Flash
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=selection_prompt
    ).text
    
    try:
        story_index = int(response.strip()) - 1
        chosen_story = selection_posts[story_index]
        
        # Save this story ID to avoid reusing
        with open(cache_file, 'a') as f:
            f.write(f"{chosen_story.id}\n")
            
        print(f"Selected story: '{chosen_story.title}' from r/{chosen_story.subreddit.display_name}")
    except (ValueError, IndexError) as e:
        # Fallback if parsing fails
        print(f"Selection error: {str(e)}. Choosing random story.")
        chosen_story = random.choice(selection_posts)

    # Create enhanced podcast-style prompt with intro and outro
    enhancement_prompt = """
    Transform this horror story into a compelling podcast episode for "The Withering Club" podcast.
    Create a complete narrative with a clear beginning, middle, and end, following this structure:

    1. PODCAST INTRO (30-45 seconds):
    "Welcome to The Withering Club, where we explore the darkest corners of human experience. I'm your host, [Host Name]. Before we begin tonight's story, remember that the shadows you see might be watching back. Now, dim the lights and prepare yourself for tonight's tale..."

    2. STORY HOOK (30 seconds):
    - Start with a powerful, intriguing hook related to the story's theme
    - Use a universal truth or historical fact that connects to the story
    - Create immediate tension or curiosity

    3. MAIN NARRATIVE:
    - BEGINNING: Establish characters, setting, and the initial situation clearly
    - MIDDLE: Develop escalating tension and conflict with vivid sensory details
    - END: Provide a satisfying conclusion that resolves the main conflict while potentially leaving some mystery

    4. PODCAST OUTRO (30-45 seconds):
    "That concludes tonight's tale from The Withering Club. If this story kept you up at night, remember to like, share, and subscribe to join our growing community of darkness seekers. Visit our website at witheringclub.com for exclusive content and merchandise. Until next time, remember... the best stories are the ones that follow you home. Sleep well, if you can."

    IMPORTANT GUIDELINES:
    - Use conversational, intimate tone throughout
    - Add atmospheric descriptions and sensory details
    - Build tension progressively
    - Include psychological elements
    - Remove any Reddit-specific references
    - Create vivid imagery through words
    - Ensure the narrative flows naturally between sections
    - Make the beginning, middle and end clearly distinguishable
    - Replace [Host Name] with a fictional host name that fits a horror podcast

Original Story: {content}
    Original Title: {title}
    Source: r/{subreddit}

    Rewrite as a complete Withering Club podcast episode that grips listeners from start to finish.
    """.format(
        content=chosen_story.selftext[:7000],
        title=chosen_story.title,
        subreddit=chosen_story.subreddit.display_name
    )

    # Get enhanced story from Gemini Flash
    enhanced_story = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=enhancement_prompt
    ).text

    # Clean up the enhanced story to remove any markdown or formatting
    enhanced_story = re.sub(r'#.*?\n', '', enhanced_story)  # Remove headers
    enhanced_story = re.sub(r'\*\*.*?\*\*', '', enhanced_story)  # Remove bold
    enhanced_story = re.sub(r'\*.*?\*', '', enhanced_story)  # Remove italics
    enhanced_story = re.sub(r'\n\s*\n\s*\n', '\n\n', enhanced_story)  # Fix spacing
    
    return {
        'title': chosen_story.title,
        'original': chosen_story.selftext,
        'enhanced': enhanced_story.strip(),
        'subreddit': chosen_story.subreddit.display_name,
        'story_id': chosen_story.id
    }


story_data = fetch_and_enhance_nosleep_story() # The indentation of this line was fixed to align with the function definition.
print(story_data['enhanced'])



# ===== CELL 7: VOICE OVER SCRIPT GENERATION =====
def generate_voice_over_script(story_text):
    """Generate a voice-over script from the enhanced story"""
    # This function can be expanded to include specific formatting for voice-over
    return story_text

# Example usage:
voice_over_script = generate_voice_over_script(story_data['enhanced']) # Removed the extra space at the beginning of the line.
print(voice_over_script)



# ===== CELL 8: AUTOMATED VOICE OVER GENERATION =====
import numpy as np
import soundfile as sf
from kokoro import KPipeline
import os

def generate_horror_audio(story_text, output_dir="audio_output"):
    """Generate professional horror narration audio"""
    os.makedirs(output_dir, exist_ok=True)

    # Initialize pipeline with horror-optimized settings
    pipeline = KPipeline(lang_code='a')

    # Professional voice parameters for horror narration
    generator = pipeline(
        story_text,
        voice='af_bella',  # Special horror-optimized voice
        speed=0.85,             # Slower pace for tension building


    )

    # Generate and concatenate audio segments
    audio_data = np.concatenate([audio.numpy() for _, _, audio in generator])

    # Save high-quality audio file
    output_path = os.path.join(output_dir, "narration.wav")
    sf.write(output_path, audio_data, 24000)

    print(f"🔊 Professional horror narration generated: {output_path}")
    return output_path

# Example usage:
#audio_path = generate_horror_audio(story_data['enhanced'])

# ===== CELL 9: SUBTITLE GENERATION =====
from transformers import pipeline
import whisper
from whisper.utils import WriteSRT

def generate_subtitles(audio_path, output_dir="subtitles"):
    """Generate subtitles using Whisper large-v3 model"""
    os.makedirs(output_dir, exist_ok=True)

    # Load Whisper model (optimized for horror narration)
    model = whisper.load_model("base")  # Use 'small' for better accuracy

    # Transcribe with timing info
    result = model.transcribe(audio_path,
                            verbose=False,
                            word_timestamps=True,
                            fp16=torch.cuda.is_available())

    # Save SRT file
    srt_path = os.path.join(output_dir, "subtitles.srt")
    with open(srt_path, "w", encoding="utf-8") as srt_file:
        writer = WriteSRT(output_dir)
        writer.write_result(result, srt_file)

    print(f"📜 Subtitles generated: {srt_path}")
    return srt_path

# Example usage:
#srt_path = generate_subtitles(audio_path)

# ===== CELL 9.1: ENHANCED SCENE DESCRIPTION GENERATION =====
def parse_srt_timestamps(srt_path):
    """Parse SRT file and extract timestamps with text"""
    segments = []
    current_segment = {}
    
    with open(srt_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.isdigit():  # Segment number
            if current_segment:
                segments.append(current_segment)
                current_segment = {}
            
            i += 1
            # Parse timestamp line
            timestamp_line = lines[i].strip()
            start_time, end_time = timestamp_line.split(' --> ')
            
            i += 1
            # Get text (may be multiple lines)
            text = []
            while i < len(lines) and lines[i].strip():
                text.append(lines[i].strip())
                i += 1
                
            current_segment = {
                'start_time': start_time,
                'end_time': end_time,
                'text': ' '.join(text)
            }
        i += 1
        
    if current_segment:
        segments.append(current_segment)
        
    return segments

def generate_scene_descriptions(srt_path, delay_seconds=2):
    """Generate cinematic scene descriptions based on subtitle segments"""
    segments = parse_srt_timestamps(srt_path)
    scene_descriptions = []
    
    chunk_size = 2
    max_retries = 3
    
    for i in range(0, len(segments) - chunk_size + 1, chunk_size):
        chunk = segments[i:i + chunk_size]
        combined_text = ' '.join([seg['text'] for seg in chunk])
        
        # Enhanced prompt focusing on visual storytelling and scenario creation
        prompt = f"""
        You are a horror film director. Create a vivid, cinematic scene description for this segment of narration.
        
        NARRATION: "{combined_text}"
        
        Imagine this as a specific moment in a horror film. Describe:
        1. The exact visual scenario that would be filmed (not abstract concepts)
        2. Characters' positions, expressions, and actions
        3. Setting details including lighting, weather, and environment
        4. Camera angle and framing (close-up, wide shot, etc.)
        5. Color palette and visual tone
        
        IMPORTANT:
        - Describe a SINGLE, SPECIFIC moment that could be photographed
        - Focus on what the VIEWER SEES, not what characters think or feel
        - Include specific visual details that create atmosphere
        - Avoid vague descriptions - be concrete and filmable
        - Write in present tense as if describing a film frame
        
        Example: "A woman stands in her dimly lit kitchen, gripping a bloodstained knife. Her face is illuminated only by moonlight streaming through venetian blinds, casting striped shadows across her vengeful expression. In the background, shadowy figures can be seen through a doorway, unaware of her presence. The camera frames her in a low-angle shot, emphasizing her newfound power."
        
        Return ONLY the scene description, no explanations or formatting.
        """
        
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                ).text
                
                # Clean up the response
                cleaned_response = (response
                    .replace('**', '')
                    .replace('Scene:', '')
                    .replace('Description:', '')
                    .strip())
                
                scene_descriptions.append({
                    'start_time': chunk[0]['start_time'],
                    'end_time': chunk[-1]['end_time'],
                    'description': cleaned_response
                })
                
                print(f"Generated scene {len(scene_descriptions)}/{(len(segments) - chunk_size + 1)//chunk_size + 1}")
                time.sleep(delay_seconds)
                break
                
            except Exception as e:
                if "429" in str(e):  # Rate limit error
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * delay_seconds * 2
                        print(f"Rate limit hit, waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        print(f"Failed after {max_retries} attempts, using fallback description")
                        fallback_desc = "A dimly lit room with shadows stretching across the walls. A figure stands motionless, their face obscured by darkness as moonlight filters through a nearby window."
                        scene_descriptions.append({
                            'start_time': chunk[0]['start_time'],
                            'end_time': chunk[-1]['end_time'],
                            'description': fallback_desc
                        })
                else:
                    print(f"Error generating scene {len(scene_descriptions) + 1}: {str(e)}")
                    break
    
    return scene_descriptions


# ===== CELL 10: ENHANCED IMAGE PROMPT GENERATION =====

# Style guidance for different visual approaches
STYLE_GUIDANCE = {
    "realistic": "photorealistic, intricate details, natural lighting, cinematic photography, 8k resolution, dramatic composition",
    "cinematic": "cinematic composition, dramatic lighting, film grain, anamorphic lens effect, professional cinematography, color grading, depth of field",
    "artistic": "digital art, stylized, vibrant colors, dramatic composition, concept art, trending on artstation, by Greg Rutkowski and Zdzisław Beksiński",
    "neutral": "balanced composition, masterful photography, perfect exposure, selective focus, attention-grabbing depth of field, highly atmospheric"
}

def enhance_prompt(prompt):
    """Add standard enhancement terms to a prompt"""
    return f"{prompt}, highly detailed, cinematic lighting, atmospheric, 8k resolution"

def generate_image_prompts(scene_descriptions, style="cinematic", delay_seconds=3):
    """Generate detailed Stable Diffusion prompts from scene descriptions"""
    prompts = []
    style_desc = STYLE_GUIDANCE.get(style, STYLE_GUIDANCE["cinematic"])
    max_retries = 3
    
    print(f"\nGenerating {len(scene_descriptions)} image prompts...")
    
    for i, scene in enumerate(scene_descriptions):
        prompt_template = f"""
        You are a professional concept artist for horror films. Create a detailed image prompt for Stable Diffusion XL based on this scene description.
        
        SCENE DESCRIPTION: "{scene['description']}"
        
        Your task is to translate this scene into a precise, visual prompt that will generate a striking horror image.
        
        Follow these requirements:
        1. Start with the main subject and their action (e.g., "A pale woman clutching a bloodied photograph")
        2. Describe the exact setting with specific details (e.g., "in an abandoned Victorian nursery with peeling wallpaper")
        3. Specify lighting, atmosphere, and color palette (e.g., "lit only by a single candle, casting long shadows, desaturated blue tones")
        4. Include camera perspective and framing (e.g., "extreme close-up shot, shallow depth of field")
        5. Add these style elements: {style_desc}
        
        IMPORTANT:
        - Be extremely specific and visual - describe exactly what should appear in the image
        - Focus on a single, powerful moment that tells a story
        - Include precise details about expressions, positioning, and environment
        - Use strong visual language that creates mood and atmosphere
        - Keep the prompt under 400 characters but dense with visual information
        
        Return ONLY the prompt text with no explanations or formatting.
        """
        
        # Attempt to generate prompt with retries and delay
        for attempt in range(max_retries):
            try:
                # Generate prompt using Gemini
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt_template
                ).text
                
                # Enhance the prompt with standard terms
                enhanced_prompt = enhance_prompt(response.strip())
                
                prompts.append({
                    'timing': (scene['start_time'], scene['end_time']),
                    'prompt': enhanced_prompt,
                    'original_description': scene['description']  # Store original for reference
                })
                
                print(f"Generated prompt {i+1}/{len(scene_descriptions)}")
                time.sleep(delay_seconds)  # Add delay between requests
                break
                
            except Exception as e:
                if "429" in str(e):  # Resource exhausted error
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * delay_seconds * 2  # Exponential backoff
                        print(f"Rate limit hit, waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        print(f"Failed after {max_retries} attempts, using fallback prompt")
                        # Use a fallback prompt based on scene description
                        fallback_prompt = f"Horror scene: {scene['description'][:100]}, dark atmosphere, cinematic lighting, film grain"
                        prompts.append({
                            'timing': (scene['start_time'], scene['end_time']),
                            'prompt': enhance_prompt(fallback_prompt),
                            'original_description': scene['description']
                        })
                else:
                    print(f"Error generating prompt {i+1}: {str(e)}")
                    break
    
    return prompts


# ===== CELL 11: STABLE DIFFUSION INITIALIZATION      =====


from diffusers import StableDiffusionXLPipeline
import torch

def initialize_stable_diffusion():
    """Initialize the Stable Diffusion XL model"""
    global sd_pipeline

    # Load Stable Diffusion XL pipeline
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    sd_pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True
    )

    # Enable attention slicing for better memory efficiency
    sd_pipeline.enable_attention_slicing()

    # Move to GPU if available
    if torch.cuda.is_available():
        sd_pipeline = sd_pipeline.to("cuda")

    print("✅ Stable Diffusion XL model initialized")
    return sd_pipeline


#initialize_stable_diffusion()

# ===== CELL 12: ENHANCED STABLE DIFFUSION IMAGE GENERATION =====

from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import torch
import os
import time
import random

def auto_generate_image(prompt):
    """Generate high-quality cinematic image with optimized SDXL settings"""
    global sd_pipeline

    # Ensure pipeline is initialized
    if sd_pipeline is None:
        print("Initializing Stable Diffusion pipeline...")
        initialize_stable_diffusion()
    
    # Set optimal scheduler for SDXL
    sd_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        sd_pipeline.scheduler.config,
        algorithm_type="sde-dpmsolver++",
        use_karras_sigmas=True
    )

    # Create a more refined negative prompt based on best practices
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
    
    # Optimal inference parameters based on SDXL guide
    image = sd_pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=40,     # Higher step count for better quality
        guidance_scale=7.5,         # Optimal CFG value for SDXL
        generator=torch_generator,
        output_type="pil"
    ).images[0]

    print(f"Image generated with seed: {seed}")
    return image

def generate_story_images(image_prompts, output_dir="auto_images"):
    """Generate high-quality images from prompts with advanced settings"""
    # Auto-create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate images with progress bar
    image_paths = []
    print(f"\nGenerating {len(image_prompts)} cinematic images...")
    
    for idx, prompt_data in enumerate(image_prompts, 1):
        output_path = os.path.join(output_dir, f"scene_{idx:03d}.png")
        
        # Generate image with multiple attempts if needed
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Extract the original prompt without enhancements to avoid redundancy
                base_prompt = prompt_data['prompt']
                
                # Create a cinematic prompt with optimal structure
                cinematic_prompt = (
                    f"{base_prompt}, cinematic lighting, dramatic composition, "
                    f"professional photography, film grain, anamorphic lens, "
                    f"depth of field, 8k resolution, hyperdetailed, masterpiece"
                )
                
                # Generate the image
                image = auto_generate_image(cinematic_prompt)
                
                # Save in high quality
                image.save(output_path, format="PNG", quality=100)
                image_paths.append(output_path)
                
                print(f"Generated image {idx}/{len(image_prompts)} (Attempt {attempt + 1})")
                break
            except Exception as e:
                if attempt == max_attempts - 1:
                    print(f"Failed to generate image {idx} after {max_attempts} attempts: {str(e)}")
                else:
                    print(f"Attempt {attempt + 1} failed, retrying...")
                    time.sleep(2)
    
    print(f"\nSuccessfully generated {len(image_paths)} cinematic images")
    return image_paths

# ===== CELL 13: ENHANCED CINEMATIC VIDEO GENERATION =====

import os
import re
import gc
import numpy as np
from moviepy.editor import *
from moviepy.video.compositing.transitions import crossfadein
from moviepy.video.tools.subtitles import SubtitlesClip
from typing import List, Optional
import sys
import contextlib
import random

# Configure ImageMagick
try:
    from moviepy.config import change_settings
    change_settings({"IMAGEMAGICK_BINARY": "convert"})
except:
    possible_paths = [
        "C:\\Program Files\\ImageMagick-7.0.10-Q16\\convert.exe",
        "C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\convert.exe",
        "/usr/bin/convert",
        "convert"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            change_settings({"IMAGEMAGICK_BINARY": path})
            break

def db_to_amplitude(db: float) -> float:
    """Convert decibels to amplitude ratio"""
    return 10 ** (db / 20)

def convert_timestamp_to_seconds(timestamp):
    """Convert SRT timestamp to seconds"""
    hours, minutes, seconds = timestamp.replace(',', '.').split(':')
    return float(hours) * 3600 + float(minutes) * 60 + float(seconds)

# Constants for optimization
BATCH_SIZE = 2
MAX_DIMENSION = 1920
JPEG_QUALITY = 85
TRANSITION_DURATION = 1.0
CINEMATIC_RATIO = 2.35  # Cinematic aspect ratio (21:9)

@contextlib.contextmanager
def suppress_output():
    """Suppress output temporarily."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def add_cinematic_black_bars(clip):
    """Add cinematic black bars to create widescreen look"""
    # Calculate the height of black bars to achieve cinematic aspect ratio
    original_height = clip.h
    original_width = clip.w
    target_height = int(original_width / CINEMATIC_RATIO)
    bar_height = (original_height - target_height) // 2
    
    # Create black bars
    if bar_height > 0:
        # Create a mask for the visible area
        mask = ColorClip(size=(original_width, target_height),
                         color=(255, 255, 255))
        mask = mask.set_position(('center', 'center')).set_duration(clip.duration)

        # Apply the mask to create letterboxed effect
        return CompositeVideoClip([
            ColorClip(size=(original_width, original_height), color=(0, 0, 0)).set_duration(clip.duration),
            clip.set_position(('center', 'center')).set_mask(mask)
        ])
    return clip

def create_final_video(image_prompts, image_paths, audio_path, title, srt_path=None):
    """Create cinematic video with dust overlay, black bars, and subtle image movements"""
    try:
        print("Starting enhanced cinematic video creation...")
        
        # Create output directory if it doesn't exist
        output_dir = "/content/output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Get audio duration
        audio = AudioFileClip(audio_path)
        total_duration = audio.duration
        
        # Check for dust overlay video
        dust_overlay = None
        dust_overlay_path = "dustoverlay.mov"
        if os.path.exists(dust_overlay_path):
            try:
                print("Loading dust overlay effect...")
                dust_overlay = VideoFileClip(dust_overlay_path, has_mask=True)
                # Loop if needed to match total duration
                if dust_overlay.duration < total_duration:
                    dust_overlay = vfx.loop(dust_overlay, duration=total_duration)
                else:
                    dust_overlay = dust_overlay.subclip(0, total_duration)
            except Exception as e:
                print(f"Warning: Could not load dust overlay: {str(e)}")
                dust_overlay = None
        else:
            print("Dust overlay file not found, continuing without it...")

        # Create clips from images with their specific timings and subtle movements
        video_clips = []
        
        for i, (prompt_data, img_path) in enumerate(zip(image_prompts, image_paths)):
            if not os.path.exists(img_path):
                continue

            # Get timing from prompt data
            start_time = convert_timestamp_to_seconds(prompt_data['timing'][0])
            end_time = convert_timestamp_to_seconds(prompt_data['timing'][1])
            duration = end_time - start_time
            
            # Add random subtle tilt/rotation to image
            tilt_angle = random.uniform(-2.0, 2.0)  # Random tilt between -2 and 2 degrees
            zoom_factor = random.uniform(1.02, 1.08)  # Random zoom between 2-8%
            
            # Create clip with subtle zoom and rotation
            clip = (ImageClip(img_path)
                   .set_duration(duration)
                   .set_start(start_time)
                   .resize(lambda t: zoom_factor + (0.1 * t/duration))  # Combine base zoom with gradual zoom
                   .rotate(lambda t: tilt_angle, expand=False)  # Apply subtle tilt
                   .set_position('center'))
            
            video_clips.append(clip)

        # Add transitions between clips
        final_clips = []
        for i, clip in enumerate(video_clips):
            if i > 0:
                # Add crossfade with previous clip
                clip = clip.crossfadein(1.0)
                final_clips.append(clip)

        # Combine all clips
        print("Combining video clips...")
        video = CompositeVideoClip(final_clips).resize(width=1920, height=1080)

        # Add dust overlay if available (using screen blend mode)
        if dust_overlay is not None:
            print("Applying dust overlay effect...")
            # Resize overlay to match video dimensions
            dust_overlay = dust_overlay.resize(video.size)
            # Apply screen blend mode
            video = CompositeVideoClip([
                video,
                dust_overlay.set_blend_mode("screen")
            ])

        # Add cinematic black bars
        print("Adding cinematic letterboxing...")
        video = add_cinematic_black_bars(video)

        # Add subtitles if available
        if srt_path and os.path.exists(srt_path):
            try:
                # First try to download and use the Anton font
                try:
                    import requests
                    import tempfile

                    # Download Anton font
                    font_url = "https://fonts.gstatic.com/s/anton/v23/1Ptgg87LROyAm3Kz-C8.woff2"
                    font_path = os.path.join(tempfile.gettempdir(), "Anton-Regular.woff2")

                    # Download the font file if it doesn't exist
                    if not os.path.exists(font_path):
                        print("Downloading Anton font...")
                        response = requests.get(font_url)
                        with open(font_path, 'wb') as f:
                            f.write(response.content)
                        print("Font downloaded successfully")

                    # Use the downloaded font
                    subtitle_font = font_path
                    print("Using Anton font for subtitles")
                except Exception as e:
                    print(f"Could not download Anton font: {str(e)}, using default font")
                    subtitle_font = 'Arial-Bold'

                # Create subtitle generator with the Anton font
                generator = lambda txt: TextClip(
                    txt,
                    font=subtitle_font,
                    fontsize=54,  # Larger size for Anton font
                    color='white',
                    stroke_color='black',
                    stroke_width=3,  # Thicker stroke for better visibility
                    method='caption',
                    size=(video.w * 0.7, None),  # Wider text area for Anton
                    align='center'
                )
                
                subs = SubtitlesClip(srt_path, generator)
                video = CompositeVideoClip([
                    video,
                    subs.set_position(('center', 'bottom'))  # Position at bottom for better visibility
                ])
            except Exception as e:
                print(f"Warning: Could not add subtitles: {str(e)}")

            # Add background music if available
            if os.path.exists("/content/ambient.mp3"):
                bg_music = (AudioFileClip("/content/ambient.mp3")
                           .volumex(db_to_amplitude(BG_MUSIC_DB)))
                
                if bg_music.duration < total_duration:
                    bg_music = afx.audio_loop(bg_music, duration=total_duration)
                else:
                    bg_music = bg_music.subclip(0, total_duration)
                
                final_audio = CompositeAudioClip([audio, bg_music])
            else:
                final_audio = audio

        video = video.set_audio(final_audio)

        # Render final video
        print("Rendering final cinematic video...")
        output_file = os.path.join(output_dir, f"{title}.mp4")
        video.write_videofile(
            output_file,
            fps=24,
            codec='libx264',
            audio_codec='aac',
            bitrate='6000k',  # Higher bitrate for better quality
            threads=4,
            preset='medium',  # Better quality preset
            ffmpeg_params=['-crf', '18']  # Higher quality CRF value
        )

        # Cleanup
        video.close()
        audio.close()
        if dust_overlay is not None:
            dust_overlay.close()
        
        print("Cinematic video creation completed successfully.")
        return output_file

    except Exception as e:
        print(f"Error in video creation: {str(e)}")
        return None


# ===== CELL 14: COMPLETE PIPELINE EXECUTION =====

from google.colab import drive
import os
import time
from datetime import datetime

def mount_google_drive():
    """Mount Google Drive if not already mounted"""
    if not os.path.exists('/content/drive'):
        print("Mounting Google Drive...")
        drive.mount('/content/drive')
    else:
        print("Google Drive already mounted")

def create_output_folders():
    """Create necessary output folders in Google Drive"""
    base_path = '/content/drive/MyDrive/HorrorStoryAI'
    folders = ['videos', 'images', 'audio', 'subtitles']
    
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
    
    return base_path

def run_complete_pipeline():
    """Execute the complete story-to-video pipeline"""
    try:
        print("Starting complete horror story pipeline...")
        start_time = time.time()
        
        # Mount Google Drive
        mount_google_drive()
        base_path = create_output_folders()
        
        # Generate timestamp for unique folder names
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_folder = os.path.join(base_path, f"story_{timestamp}")
        os.makedirs(project_folder, exist_ok=True)
        
        # 1. Fetch and enhance story
        print("\n1. Fetching and enhancing story...")
        story_data = fetch_and_enhance_nosleep_story()
        
        # Save story text
        with open(os.path.join(project_folder, "story.txt"), 'w', encoding='utf-8') as f:
            f.write(f"Title: {story_data['title']}\n\n")
            f.write(f"Enhanced Story:\n{story_data['enhanced']}")
        
        # 2. Generate voice over script
        print("\n2. Generating voice-over script...")
        voice_over_script = generate_voice_over_script(story_data['enhanced'])
        
        # 3. Generate audio narration
        print("\n3. Generating audio narration...")
        audio_path = generate_horror_audio(voice_over_script)
        
        # 4. Generate subtitles
        print("\n4. Generating subtitles...")
        srt_path = generate_subtitles(audio_path)
        
        # 5. Generate scene descriptions
        print("\n5. Generating scene descriptions...")
        scene_descriptions = generate_scene_descriptions(srt_path)
        
        # 5.5 Generate ambient soundscape
        print("\n5.5 Generating ambient sound design...")
        ambient_path = generate_ambient_soundscape(
            scene_descriptions=scene_descriptions,
            audio_duration=AudioFileClip(audio_path).duration
        )
        
        # 6. Generate image prompts
        print("\n6. Generating image prompts...")
        image_prompts = generate_image_prompts(scene_descriptions)
        
        # 7. Initialize Stable Diffusion
        print("\n7. Initializing Stable Diffusion...")
        initialize_stable_diffusion()
        
        # 8. Generate images
        print("\n8. Generating images...")
        image_paths = generate_story_images(image_prompts)
        
        # 9. Create final video with ambient sound
        print("\n9. Creating final video...")
        video_path = create_final_video(
            image_prompts=image_prompts,
            image_paths=image_paths,
            audio_path=audio_path,
            title=f"horror_story_{timestamp}",
            srt_path=srt_path,
            ambient_path=ambient_path
        )

        # 10. Save all outputs to Google Drive
        print("\n10. Saving outputs to Google Drive...")
        
        # Copy files to appropriate folders
        import shutil
        
        # Save video
        if video_path and os.path.exists(video_path):
            video_dest = os.path.join(base_path, 'videos', os.path.basename(video_path))
            shutil.copy2(video_path, video_dest)
        
        # Save images
        for img_path in image_paths:
            if os.path.exists(img_path):
                img_dest = os.path.join(base_path, 'images', os.path.basename(img_path))
                shutil.copy2(img_path, img_dest)
        
        # Save audio
        if os.path.exists(audio_path):
            audio_dest = os.path.join(base_path, 'audio', os.path.basename(audio_path))
            shutil.copy2(audio_path, audio_dest)
        
        # Save subtitles
        if os.path.exists(srt_path):
            srt_dest = os.path.join(base_path, 'subtitles', os.path.basename(srt_path))
            shutil.copy2(srt_path, srt_dest)
        
        # Calculate total time
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\nComplete pipeline executed successfully in {total_time/60:.2f} minutes!")
        print(f"All outputs saved to: {project_folder}")
        
        # Return paths for reference
        return {
            'project_folder': project_folder,
            'video_path': video_path,
            'image_paths': image_paths,
            'audio_path': audio_path,
            'srt_path': srt_path,
            'story_data': story_data
        }
        
    except Exception as e:
        print(f"Error in pipeline execution: {str(e)}")
        return None

# Execute the complete pipeline
if __name__ == "__main__":
    results = run_complete_pipeline()
    
    if results and results['video_path'] and os.path.exists(results['video_path']):
        # Display the final video
        from IPython.display import HTML
        from base64 import b64encode

        mp4 = open(results['video_path'], 'rb').read()
        data_url = f"data:video/mp4;base64,{b64encode(mp4).decode()}"
        display(HTML(f"""
        <video width="640" height="360" controls>
            <source src="{data_url}" type="video/mp4">
        </video>
        """))

# ===== CELL 8.5: AMBIENT SOUND DESIGN GENERATION =====

import numpy as np
import soundfile as sf
from pydub import AudioSegment
import random
import os
import re
from typing import List, Dict, Tuple

class AmbientSoundDesigner:
    """Creates contextual ambient sound design for horror stories"""
    
    def __init__(self, sound_library_path="sound_effects"):
        """Initialize with path to sound effect library"""
        self.sound_library_path = sound_library_path
        self.sound_categories = {
            'indoor': ['creaking', 'footsteps', 'door', 'clock', 'breathing', 'whisper'],
            'outdoor': ['wind', 'rain', 'thunder', 'leaves', 'branches', 'animals'],
            'tension': ['heartbeat', 'drone', 'strings', 'pulse', 'rumble', 'static'],
            'horror': ['scream', 'whisper', 'laugh', 'growl', 'scratch', 'thump']
        }
        
        # Create sound library directory if it doesn't exist
        os.makedirs(self.sound_library_path, exist_ok=True)
        
        # Download basic sound effects if library is empty
        if not os.listdir(self.sound_library_path):
            self._download_basic_sound_library()
    
    def _download_basic_sound_library(self):
        """Download a basic set of sound effects from Freesound or similar"""
        try:
            import requests
            
            # Basic sound effects URLs (replace with actual URLs)
            sound_urls = {
                'indoor_creaking.wav': 'https://example.com/creaking.wav',
                'indoor_footsteps.wav': 'https://example.com/footsteps.wav',
                'outdoor_wind.wav': 'https://example.com/wind.wav',
                'tension_drone.wav': 'https://example.com/drone.wav',
                'horror_whisper.wav': 'https://example.com/whisper.wav',
                # Add more sound effects as needed
            }
            
            print("Downloading basic sound library...")
            for filename, url in sound_urls.items():
                try:
                    response = requests.get(url)
                    if response.status_code == 200:
                        with open(os.path.join(self.sound_library_path, filename), 'wb') as f:
                            f.write(response.content)
                        print(f"Downloaded {filename}")
                except Exception as e:
                    print(f"Failed to download {filename}: {str(e)}")
                    
        except ImportError:
            print("Requests library not available. Please install sound effects manually.")
    
    def analyze_scene(self, scene_description: str) -> Dict[str, float]:
        """Analyze scene description to determine appropriate sound categories and intensities"""
        scene_scores = {category: 0.0 for category in self.sound_categories}
        
        # Keywords that indicate different environments and moods
        keywords = {
            'indoor': ['room', 'house', 'building', 'inside', 'hallway', 'corridor', 'bedroom', 
                      'kitchen', 'basement', 'attic', 'stairs', 'floor'],
            'outdoor': ['forest', 'woods', 'outside', 'street', 'road', 'field', 'sky', 
                       'rain', 'storm', 'wind', 'night', 'day', 'sun', 'moon'],
            'tension': ['fear', 'anxiety', 'nervous', 'tense', 'suspense', 'dread', 
                       'worry', 'panic', 'terror', 'horror', 'afraid'],
            'horror': ['blood', 'scream', 'death', 'monster', 'creature', 'ghost', 
                      'shadow', 'dark', 'evil', 'demon', 'supernatural', 'haunted']
        }
        
        # Calculate scores based on keyword presence
        for category, words in keywords.items():
            for word in words:
                if re.search(r'\b' + word + r'\b', scene_description.lower()):
                    scene_scores[category] += 0.2  # Increase score for each keyword found
        
        # Normalize scores to range 0-1
        max_score = max(scene_scores.values()) if max(scene_scores.values()) > 0 else 1.0
        for category in scene_scores:
            scene_scores[category] = min(scene_scores[category] / max_score, 1.0)
            
        # Ensure at least some ambient sound
        if all(score < 0.2 for score in scene_scores.values()):
            scene_scores['tension'] = 0.3
            
        return scene_scores
    
    def select_sounds(self, scene_scores: Dict[str, float], duration: float) -> List[Dict]:
        """Select appropriate sound effects based on scene analysis"""
        selected_sounds = []
        
        # Get available sound files
        available_sounds = {}
        for category in self.sound_categories:
            available_sounds[category] = []
            for sound_type in self.sound_categories[category]:
                pattern = f"{category}_{sound_type}*.wav"
                matching_files = []
                for file in os.listdir(self.sound_library_path):
                    if file.startswith(f"{category}_{sound_type}") and file.endswith(".wav"):
                        matching_files.append(file)
                available_sounds[category].extend(matching_files)
        
        # Select sounds based on scores
        for category, score in scene_scores.items():
            if score > 0.2 and available_sounds[category]:  # Only use categories with significant scores
                num_sounds = int(score * 3)  # More sounds for higher scores
                for _ in range(min(num_sounds, len(available_sounds[category]))):
                    sound_file = random.choice(available_sounds[category])
                    volume = 0.3 + (score * 0.7)  # Volume based on score (0.3-1.0)
                    
                    selected_sounds.append({
                        'file': os.path.join(self.sound_library_path, sound_file),
                        'volume': volume,
                        'category': category,
                        'loop': category in ['tension', 'outdoor'],  # Loop background sounds
                        'random_start': category != 'tension'  # Random start time for non-tension sounds
                    })
        
        return selected_sounds
    
    def create_ambient_mix(self, scene_descriptions: List[Dict], output_path: str, total_duration: float) -> str:
        """Create a complete ambient sound mix for the entire story"""
        # Create base silent track
        base_track = AudioSegment.silent(duration=int(total_duration * 1000))
        
        # Process each scene
        for i, scene in enumerate(scene_descriptions):
            # Calculate scene timing
            start_time = convert_timestamp_to_seconds(scene['start_time'])
            end_time = convert_timestamp_to_seconds(scene['end_time'])
            scene_duration = end_time - start_time
            
            # Analyze scene and select sounds
            scene_scores = self.analyze_scene(scene['description'])
            selected_sounds = self.select_sounds(scene_scores, scene_duration)
            
            print(f"Creating ambient sound for scene {i+1}/{len(scene_descriptions)}")
            
            # Mix sounds for this scene
            for sound_info in selected_sounds:
                try:
                    # Load sound file
                    sound = AudioSegment.from_file(sound_info['file'])
                    
                    # Adjust volume
                    sound = sound - (20 - (sound_info['volume'] * 20))  # Convert to dB reduction
                    
                    # Handle looping for background sounds
                    if sound_info['loop'] and sound.duration_seconds < scene_duration:
                        loops_needed = int(scene_duration / sound.duration_seconds) + 1
                        sound = sound * loops_needed
                    
                    # Trim to scene duration
                    if sound.duration_seconds > scene_duration:
                        # Use random start point if specified
                        if sound_info['random_start'] and sound.duration_seconds > scene_duration * 1.5:
                            max_start = int((sound.duration_seconds - scene_duration) * 1000)
                            start_pos = random.randint(0, max_start)
                            sound = sound[start_pos:start_pos + int(scene_duration * 1000)]
                        else:
                            sound = sound[:int(scene_duration * 1000)]
                    
                    # Add crossfade at beginning and end (except for tension sounds)
                    if not sound_info['category'] == 'tension':
                        fade_duration = min(1000, int(sound.duration_seconds * 1000 * 0.2))
                        sound = sound.fade_in(fade_duration).fade_out(fade_duration)
                    
                    # Position in the timeline
                    position_ms = int(start_time * 1000)
                    base_track = base_track.overlay(sound, position=position_ms)
                    
                except Exception as e:
                    print(f"Error processing sound {sound_info['file']}: {str(e)}")
        
        # Export final mix
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        base_track.export(output_path, format="wav")
        print(f"Ambient sound design created: {output_path}")
        
        return output_path

def generate_ambient_soundscape(scene_descriptions, audio_duration, output_dir="audio_output"):
    """Generate ambient sound design based on scene descriptions"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize sound designer
    sound_designer = AmbientSoundDesigner()
    
    # Create ambient sound mix
    ambient_path = os.path.join(output_dir, "ambient_soundscape.wav")
    return sound_designer.create_ambient_mix(scene_descriptions, ambient_path, audio_duration)