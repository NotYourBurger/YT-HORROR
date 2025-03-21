import gradio as gr
import os
import sys
import time
from datetime import datetime
import traceback
import json
from IPython.display import display, HTML

# Import functions from pipeline.py
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
    generate_ambient_soundscape,
    HORROR_SUBREDDITS
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
    progress=gr.Progress()
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
        
        # Store user preferences in a global variable to be accessed by pipeline functions
        global user_prefs
        user_prefs = {
            'subreddits': {'value': subreddits.split(',')},
            'min_length': {'value': min_length},
            'voice_speed': {'value': voice_speed},
            'voice_selection': {'value': voice_selection},
            'video_quality': {'value': video_quality},
            'cinematic_ratio': {'value': float(cinematic_ratio)},
            'use_dust_overlay': {'value': use_dust_overlay}
        }
        
        # Update the user_prefs in the pipeline module
        import pipeline
        pipeline.user_prefs = user_prefs
        
        # 1. Fetch and enhance story (10%)
        progress(0.1, desc="Fetching and enhancing story...")
        story_data = fetch_and_enhance_nosleep_story()
        
        # Save story text
        story_path = os.path.join(project_folder, "story.txt")
        with open(story_path, 'w', encoding='utf-8') as f:
            f.write(f"Title: {story_data['title']}\n\n")
            f.write(f"Enhanced Story:\n{story_data['enhanced']}")
        
        # 2. Generate voice over script (15%)
        progress(0.15, desc="Generating voice-over script...")
        voice_over_script = generate_voice_over_script(story_data['enhanced'])
        
        # 3. Generate audio narration (30%)
        progress(0.3, desc="Generating audio narration...")
        audio_path = generate_horror_audio(voice_over_script)
        
        # 4. Generate subtitles (40%)
        progress(0.4, desc="Generating subtitles...")
        srt_path = generate_subtitles(audio_path)
        
        # 5. Generate scene descriptions (50%)
        progress(0.5, desc="Generating scene descriptions...")
        scene_descriptions = generate_scene_descriptions(srt_path)
        
        # 5.5 Generate ambient soundscape (60%)
        progress(0.6, desc="Generating ambient sound design...")
        from moviepy.editor import AudioFileClip
        ambient_path = generate_ambient_soundscape(
            scene_descriptions=scene_descriptions,
            audio_duration=AudioFileClip(audio_path).duration
        )
        
        # 6. Generate image prompts (70%)
        progress(0.7, desc="Generating image prompts...")
        image_prompts = generate_image_prompts(scene_descriptions)
        
        # 7. Initialize Stable Diffusion (75%)
        progress(0.75, desc="Initializing Stable Diffusion...")
        initialize_stable_diffusion()
        
        # 8. Generate images (85%)
        progress(0.85, desc="Generating images...")
        image_paths = generate_story_images(image_prompts)
        
        # 9. Create final video with ambient sound (95%)
        progress(0.95, desc="Creating final video...")
        video_path = create_final_video(
            image_prompts=image_prompts,
            image_paths=image_paths,
            audio_path=audio_path,
            title=f"horror_story_{timestamp}",
            srt_path=srt_path,
            ambient_path=ambient_path
        )
        
        # 10. Complete (100%)
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

# Create the Gradio interface
def create_ui():
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
            generate_btn.click(
                fn=run_pipeline,
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
                api_name="generate",
                postprocess=display_results
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

# Launch the app
if __name__ == "__main__":
    app = create_ui()
    app.launch(share=True) 