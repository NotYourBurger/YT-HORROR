import os
import sys
import subprocess
import time
from datetime import datetime
import traceback
import json

# Install required packages first
def install_dependencies():
    print("Installing required dependencies...")
    
    # List of packages to install
    packages = [
        "praw google-generativeai numpy soundfile pillow tqdm diffusers transformers accelerate ftfy safetensors",
        "google-colab",
        "git+https://github.com/openai/whisper.git",
        "kokoro",
        "moviepy",
        "gradio>=4.0.0"
    ]
    
    # Install Python packages
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except Exception as e:
            print(f"Warning: Could not install {package}: {str(e)}")
    
    # Install system dependencies
    system_commands = [
        "apt-get update",
        "apt-get install -y ffmpeg",
        "apt install imagemagick",
        "apt install libmagick++-dev",
        "apt update && apt install -y ffmpeg fonts-dejavu"
    ]
    
    for cmd in system_commands:
        try:
            subprocess.check_call(cmd.split())
        except Exception as e:
            print(f"Warning: Could not execute '{cmd}': {str(e)}")
    
    # Fix ImageMagick policy
    try:
        subprocess.check_call("cat /etc/ImageMagick-6/policy.xml | sed 's/none/read,write/g'> /tmp/policy.xml", shell=True)
        subprocess.check_call("sudo cp /tmp/policy.xml /etc/ImageMagick-6/policy.xml", shell=True)
    except Exception as e:
        print(f"Warning: Could not update ImageMagick policy: {str(e)}")
    
    print("Dependencies installed successfully!")

# Create the Gradio interface first, before importing pipeline
def create_ui():
    import gradio as gr
    
    with gr.Blocks(title="AI Horror Story Generator") as app:
        gr.Markdown("# AI Horror Story Generator")
        gr.Markdown("Generate cinematic horror videos with AI-powered narration, images, and audio")
        
        with gr.Tab("Generate Horror Story"):
            with gr.Row():
                with gr.Column():
                    # Input parameters
                    gr.Markdown("### Story Settings")
                    subreddits = gr.Textbox(
                        label="Subreddits (comma-separated)", 
                        value="nosleep,shortscarystories,creepypasta",
                        info="Horror subreddits to fetch stories from"
                    )
                    min_length = gr.Slider(
                        minimum=500, 
                        maximum=5000, 
                        value=1000, 
                        step=500, 
                        label="Minimum Story Length"
                    )
                    
                    gr.Markdown("### Voice Settings")
                    voice_selection = gr.Dropdown(
                        choices=["af_bella", "en_joe", "en_emily"], 
                        value="af_bella", 
                        label="Narrator Voice"
                    )
                    voice_speed = gr.Slider(
                        minimum=0.5, 
                        maximum=1.2, 
                        value=0.85, 
                        step=0.05, 
                        label="Narration Speed"
                    )
                    
                    gr.Markdown("### Video Settings")
                    video_quality = gr.Dropdown(
                        choices=["6000k", "4000k", "2000k"], 
                        value="4000k", 
                        label="Video Quality"
                    )
                    cinematic_ratio = gr.Dropdown(
                        choices=[
                            ("Cinematic (2.35:1)", "2.35"),
                            ("Standard (16:9)", "1.77"),
                            ("Square (1:1)", "1.0")
                        ], 
                        value="2.35", 
                        label="Aspect Ratio"
                    )
                    use_dust_overlay = gr.Checkbox(
                        value=True, 
                        label="Add Dust Overlay Effect"
                    )
                    
                    # Run button
                    generate_btn = gr.Button("Generate Horror Story", variant="primary")
                
                with gr.Column():
                    # Output display
                    output_text = gr.Markdown("Results will appear here")
                    video_output = gr.Video(label="Generated Horror Video")
                    audio_output = gr.Audio(label="Generated Audio Narration")
            
            # Connect the button to the pipeline function
            # We'll define the actual function later after importing pipeline
            generate_btn.click(
                fn=None,  # Will be set after UI creation
                inputs=[
                    subreddits,
                    min_length,
                    voice_speed,
                    voice_selection,
                    video_quality,
                    cinematic_ratio,
                    use_dust_overlay
                ],
                outputs=[output_text, video_output, audio_output],
                api_name="generate"
            )
        
        with gr.Tab("About"):
            gr.Markdown("""
            ## AI Horror Story Generator
            
            This application automatically creates horror videos by combining:
            
            - AI-enhanced horror stories from Reddit
            - Professional-quality voice narration
            - AI-generated atmospheric images
            - Cinematic video assembly with subtitles
            
            ### How It Works
            
            1. The system fetches horror stories from Reddit
            2. AI enhances the story for audio narration
            3. Text-to-speech creates a professional voiceover
            4. AI generates thematic horror images for each scene
            5. Everything is combined into a cinematic horror video
            
            ### Credits
            
            - Uses Stable Diffusion XL for image generation
            - Powered by Google Gemini AI for story enhancement
            - Kokoro for text-to-speech narration
            """)
    
    return app

# Main execution
if __name__ == "__main__":
    # Install dependencies first
    install_dependencies()
    
    # Create the UI first (without pipeline imports)
    app = create_ui()
    
    # Now import the pipeline functions
    from pipeline import (
        fetch_and_enhance_nosleep_story,
        generate_voice_over_script,
        generate_horror_audio,
        generate_subtitles,
        generate_scene_descriptions,
        generate_image_prompts,
        initialize_stable_diffusion,
        generate_story_images,
        create_final_video,
        generate_ambient_soundscape
    )
    
    # Global variables to store intermediate results
    story_data = None
    audio_path = None
    srt_path = None
    scene_descriptions = None
    image_prompts = None
    image_paths = None
    ambient_path = None
    
    def create_output_folders():
        """Create necessary output folders"""
        folders = ['videos', 'images', 'audio', 'subtitles', 'stories']
        
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
        
        return os.getcwd()
    
    def run_pipeline(
        subreddits, 
        min_length, 
        voice_speed, 
        voice_selection, 
        video_quality, 
        cinematic_ratio, 
        use_dust_overlay,
        progress=None
    ):
        """Run the complete horror story pipeline with progress updates"""
        global story_data, audio_path, srt_path, scene_descriptions, image_prompts, image_paths, ambient_path
        
        try:
            # Create output folders
            base_path = create_output_folders()
            
            # Generate timestamp for unique folder names
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_folder = os.path.join(base_path, f"story_{timestamp}")
            os.makedirs(project_folder, exist_ok=True)
            
            # Update pipeline settings
            import pipeline
            
            # Override pipeline settings with UI values
            pipeline.HORROR_SUBREDDITS = subreddits.split(',')
            
            # Create a temporary user_prefs dictionary for functions that still use it
            pipeline.user_prefs = {
                'subreddits': {'value': subreddits.split(',')},
                'min_length': {'value': min_length},
                'voice_speed': {'value': voice_speed},
                'voice_selection': {'value': voice_selection},
                'video_quality': {'value': video_quality},
                'cinematic_ratio': {'value': float(cinematic_ratio)},
                'use_dust_overlay': {'value': use_dust_overlay}
            }
            
            # Set global CINEMATIC_RATIO in pipeline
            pipeline.CINEMATIC_RATIO = float(cinematic_ratio)
            
            # Update progress if available
            if progress:
                progress(0.1, desc="Fetching and enhancing story...")
            
            # 1. Fetch and enhance story
            story_data = fetch_and_enhance_nosleep_story()
            
            # Save story text
            story_path = os.path.join(project_folder, "story.txt")
            with open(story_path, 'w', encoding='utf-8') as f:
                f.write(f"Title: {story_data['title']}\n\n")
                f.write(f"Enhanced Story:\n{story_data['enhanced']}")
            
            if progress:
                progress(0.15, desc="Generating voice-over script...")
            
            # 2. Generate voice over script
            voice_over_script = generate_voice_over_script(story_data['enhanced'])
            
            if progress:
                progress(0.3, desc="Generating audio narration...")
            
            # 3. Generate audio narration with user-selected voice
            audio_path = generate_horror_audio(
                voice_over_script, 
                output_dir=os.path.join(project_folder, "audio")
            )
            
            if progress:
                progress(0.4, desc="Generating subtitles...")
            
            # 4. Generate subtitles
            srt_path = generate_subtitles(
                audio_path, 
                output_dir=os.path.join(project_folder, "subtitles")
            )
            
            if progress:
                progress(0.5, desc="Generating scene descriptions...")
            
            # 5. Generate scene descriptions
            scene_descriptions = generate_scene_descriptions(srt_path)
            
            if progress:
                progress(0.6, desc="Generating ambient sound design...")
            
            # 5.5 Generate ambient soundscape
            from moviepy.editor import AudioFileClip
            ambient_path = generate_ambient_soundscape(
                scene_descriptions=scene_descriptions,
                audio_duration=AudioFileClip(audio_path).duration
            )
            
            if progress:
                progress(0.7, desc="Generating image prompts...")
            
            # 6. Generate image prompts
            image_prompts = generate_image_prompts(scene_descriptions)
            
            if progress:
                progress(0.75, desc="Initializing Stable Diffusion...")
            
            # 7. Initialize Stable Diffusion
            initialize_stable_diffusion()
            
            if progress:
                progress(0.85, desc="Generating images...")
            
            # 8. Generate images
            image_paths = generate_story_images(
                image_prompts, 
                output_dir=os.path.join(project_folder, "images")
            )
            
            if progress:
                progress(0.95, desc="Creating final video...")
            
            # 9. Create final video with ambient sound
            video_path = create_final_video(
                image_prompts=image_prompts,
                image_paths=image_paths,
                audio_path=audio_path,
                title=f"horror_story_{timestamp}",
                srt_path=srt_path
            )
            
            if progress:
                progress(1.0, desc="Pipeline completed successfully!")
            
            # Return paths for display
            return {
                'story_title': story_data['title'],
                'story_text': story_data['enhanced'],
                'video_path': video_path,
                'audio_path': audio_path,
                'project_folder': project_folder
            }
            
        except Exception as e:
            error_msg = f"Error in pipeline execution: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return {"error": error_msg}
    
    def display_results(results):
        """Format results for display in the UI"""
        if isinstance(results, dict) and "error" in results:
            return f"Error: {results['error']}", None, None
        
        # Extract results
        story_title = results.get('story_title', 'Untitled')
        story_text = results.get('story_text', '')
        video_path = results.get('video_path', '')
        audio_path = results.get('audio_path', '')
        
        # Format output message
        output_msg = f"## Successfully Generated: {story_title}\n\n"
        output_msg += f"Story length: {len(story_text)} characters\n"
        
        if video_path and os.path.exists(video_path):
            output_msg += f"Video saved to: {video_path}\n"
        
        if audio_path and os.path.exists(audio_path):
            output_msg += f"Audio saved to: {audio_path}\n"
        
        # Return formatted message and media paths
        return output_msg, video_path if os.path.exists(video_path) else None, audio_path if os.path.exists(audio_path) else None
    
    # Now that we've defined the run_pipeline function, connect it to the UI
    app.queue()
    app.launch(share=True, prevent_thread_lock=True)
    
    # Get the generate button and connect it to our function
    generate_btn = [component for component in app.blocks.values() if hasattr(component, 'label') and component.label == "Generate Horror Story"][0]
    generate_btn.click(
        fn=run_pipeline,
        inputs=[
            app.blocks[component] for component in app.blocks if hasattr(app.blocks[component], 'label') and app.blocks[component].label in [
                "Subreddits (comma-separated)", "Minimum Story Length", "Narration Speed", 
                "Narrator Voice", "Video Quality", "Aspect Ratio", "Add Dust Overlay Effect"
            ]
        ],
        outputs=[
            app.blocks[component] for component in app.blocks if hasattr(app.blocks[component], 'label') and app.blocks[component].label in [
                "Results will appear here", "Generated Horror Video", "Generated Audio Narration"
            ]
        ],
        postprocess=display_results
    ) 