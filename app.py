import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import threading
import sqlite3
import json
import time
from werkzeug.utils import secure_filename

# Import pipeline components
from models.model_manager import ModelManager
from pipeline.story_fetcher import StoryFetcher
from pipeline.voice_generator import VoiceGenerator
from pipeline.subtitle_generator import SubtitleGenerator
from pipeline.scene_generator import SceneGenerator
from pipeline.image_generator import ImageGenerator
from pipeline.sound_designer import SoundDesigner
from pipeline.video_compiler import VideoCompiler

app = Flask(__name__, static_folder='./client/build', static_url_path='/')
CORS(app)

# Initialize model manager
model_manager = ModelManager(os.path.join(os.path.dirname(__file__), 'models'))

# Create a storage directory for generated content
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'storage')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Database initialization
def init_db():
    conn = sqlite3.connect('horror_app.db')
    cursor = conn.cursor()
    
    # Create tables for projects, generated content, and model status
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        date_created TEXT,
        status TEXT,
        settings TEXT,
        story_id TEXT,
        subreddit TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS generated_content (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER,
        content_type TEXT,
        file_path TEXT,
        date_created TEXT,
        FOREIGN KEY (project_id) REFERENCES projects (id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS model_status (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT,
        status TEXT,
        download_progress REAL,
        last_updated TEXT
    )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# Maintain in-memory job status
job_status = {}

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/models', methods=['GET'])
def list_models():
    """List all models and their installation status"""
    return jsonify(model_manager.list_models())

@app.route('/api/models/<model_name>/download', methods=['POST'])
def download_model(model_name):
    """Start downloading a specific model"""
    if model_name not in model_manager.get_available_models():
        return jsonify({"error": "Model not found"}), 404
    
    # Start model download in a background thread
    thread = threading.Thread(target=model_manager.download_model, args=(model_name,))
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "download_started", "model": model_name})

@app.route('/api/models/<model_name>/status', methods=['GET'])
def model_status(model_name):
    """Get the current download/installation status of a model"""
    status = model_manager.get_model_status(model_name)
    return jsonify(status)

@app.route('/api/reddit/horror_subreddits', methods=['GET'])
def get_horror_subreddits():
    """Get list of horror subreddits"""
    # These are the subreddits from the original pipeline
    subreddits = [
        "nosleep", "shortscarystories", "creepypasta", "LetsNotMeet",
        "DarkTales", "TheCrypticCompendium", "libraryofshadows",
        "scarystories", "TrueScaryStories", "HorrorStories"
    ]
    return jsonify(subreddits)

@app.route('/api/settings', methods=['POST'])
def save_settings():
    """Save user preferences"""
    settings = request.json
    
    # Connect to database
    conn = sqlite3.connect('horror_app.db')
    cursor = conn.cursor()
    
    # Store settings in the database
    cursor.execute(
        "INSERT INTO settings (settings_json, date_created) VALUES (?, datetime('now'))",
        (json.dumps(settings),)
    )
    
    conn.commit()
    settings_id = cursor.lastrowid
    conn.close()
    
    return jsonify({"id": settings_id, "settings": settings})

@app.route('/api/stories/fetch', methods=['POST'])
def fetch_story():
    """Fetch a horror story based on user preferences"""
    # Check if required models are installed
    if not model_manager.check_model_installed('gemini'):
        return jsonify({"error": "Gemini API model not installed"}), 400
    
    # Get preferences from request
    preferences = request.json
    
    # Initialize story fetcher
    fetcher = StoryFetcher(
        subreddits=preferences.get('subreddits', ['nosleep', 'shortscarystories']),
        min_length=preferences.get('min_length', 1000)
    )
    
    # Track job status
    job_id = str(time.time())
    job_status[job_id] = {"status": "running", "progress": 0, "message": "Fetching story..."}
    
    # Function to run in background thread
    def fetch_and_enhance():
        try:
            story_data = fetcher.fetch_and_enhance_story()
            
            # Save story to database
            conn = sqlite3.connect('horror_app.db')
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO projects (title, date_created, status, story_id, subreddit) VALUES (?, datetime('now'), ?, ?, ?)",
                (story_data['title'], "story_fetched", story_data['story_id'], story_data['subreddit'])
            )
            
            project_id = cursor.lastrowid
            
            # Save the story content
            story_path = os.path.join(app.config['UPLOAD_FOLDER'], f"story_{project_id}.txt")
            with open(story_path, 'w', encoding='utf-8') as f:
                f.write(f"Title: {story_data['title']}\n\n")
                f.write(f"Enhanced Story:\n{story_data['enhanced']}")
            
            cursor.execute(
                "INSERT INTO generated_content (project_id, content_type, file_path, date_created) VALUES (?, ?, ?, datetime('now'))",
                (project_id, "story", story_path)
            )
            
            conn.commit()
            conn.close()
            
            # Update job status
            job_status[job_id] = {
                "status": "completed", 
                "progress": 100, 
                "message": "Story fetched and enhanced successfully",
                "data": {
                    "project_id": project_id,
                    "title": story_data['title'],
                    "story_path": story_path
                }
            }
        except Exception as e:
            job_status[job_id] = {"status": "error", "message": str(e)}
    
    # Start background processing
    thread = threading.Thread(target=fetch_and_enhance)
    thread.daemon = True
    thread.start()
    
    return jsonify({"job_id": job_id})

@app.route('/api/job/<job_id>/status', methods=['GET'])
def get_job_status(job_id):
    """Get the status of a background job"""
    if job_id not in job_status:
        return jsonify({"error": "Job not found"}), 404
    
    return jsonify(job_status[job_id])

@app.route('/api/project/<project_id>/generate_voice', methods=['POST'])
def generate_voice(project_id):
    """Generate voice-over for a project"""
    # Check if required models are installed
    if not model_manager.check_model_installed('tts'):
        return jsonify({"error": "Text-to-speech model not installed"}), 400
    
    # Get project details from database
    conn = sqlite3.connect('horror_app.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT title FROM projects WHERE id = ?", (project_id,))
    project = cursor.fetchone()
    
    if not project:
        conn.close()
        return jsonify({"error": "Project not found"}), 404
    
    # Get the story content
    cursor.execute(
        "SELECT file_path FROM generated_content WHERE project_id = ? AND content_type = 'story'", 
        (project_id,)
    )
    result = cursor.fetchone()
    
    if not result:
        conn.close()
        return jsonify({"error": "Story content not found"}), 404
    
    story_path = result[0]
    
    # Read the story
    with open(story_path, 'r', encoding='utf-8') as f:
        story_text = f.read()
    
    # Get voice preferences
    preferences = request.json
    
    # Track job status
    job_id = str(time.time())
    job_status[job_id] = {"status": "running", "progress": 0, "message": "Generating voice-over..."}
    
    # Function to run in background thread
    def generate_voice_audio():
        try:
            # Initialize voice generator
            voice_gen = VoiceGenerator(
                voice=preferences.get('voice', 'en_emily'),
                speed=preferences.get('speed', 0.85)
            )
            
            # Extract just the enhanced story part
            enhanced_story = story_text.split("Enhanced Story:\n")[1] if "Enhanced Story:\n" in story_text else story_text
            
            # Generate voice-over
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"narration_{project_id}.wav")
            voice_gen.generate_audio(enhanced_story, audio_path)
            
            # Save to database
            conn = sqlite3.connect('horror_app.db')
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO generated_content (project_id, content_type, file_path, date_created) VALUES (?, ?, ?, datetime('now'))",
                (project_id, "audio", audio_path)
            )
            
            # Update project status
            cursor.execute(
                "UPDATE projects SET status = ? WHERE id = ?",
                ("voice_generated", project_id)
            )
            
            conn.commit()
            conn.close()
            
            # Update job status
            job_status[job_id] = {
                "status": "completed", 
                "progress": 100, 
                "message": "Voice-over generated successfully",
                "data": {
                    "project_id": project_id,
                    "audio_path": audio_path
                }
            }
        except Exception as e:
            job_status[job_id] = {"status": "error", "message": str(e)}
    
    # Start background processing
    thread = threading.Thread(target=generate_voice_audio)
    thread.daemon = True
    thread.start()
    
    return jsonify({"job_id": job_id})

@app.route('/api/project/<project_id>/generate_subtitles', methods=['POST'])
def generate_subtitles(project_id):
    """Generate subtitles for a project"""
    # Check if required models are installed
    if not model_manager.check_model_installed('whisper'):
        return jsonify({"error": "Whisper model not installed"}), 400
    
    # Get project audio file
    conn = sqlite3.connect('horror_app.db')
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT file_path FROM generated_content WHERE project_id = ? AND content_type = 'audio'", 
        (project_id,)
    )
    result = cursor.fetchone()
    
    if not result:
        conn.close()
        return jsonify({"error": "Audio content not found"}), 404
    
    audio_path = result[0]
    
    # Track job status
    job_id = str(time.time())
    job_status[job_id] = {"status": "running", "progress": 0, "message": "Generating subtitles..."}
    
    # Function to run in background thread
    def generate_audio_subtitles():
        try:
            # Initialize subtitle generator
            subtitle_gen = SubtitleGenerator()
            
            # Generate subtitles
            subtitles_path = os.path.join(app.config['UPLOAD_FOLDER'], f"subtitles_{project_id}.srt")
            subtitle_gen.generate_subtitles(audio_path, subtitles_path)
            
            # Save to database
            conn = sqlite3.connect('horror_app.db')
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO generated_content (project_id, content_type, file_path, date_created) VALUES (?, ?, ?, datetime('now'))",
                (project_id, "subtitles", subtitles_path)
            )
            
            # Update project status
            cursor.execute(
                "UPDATE projects SET status = ? WHERE id = ?",
                ("subtitles_generated", project_id)
            )
            
            conn.commit()
            conn.close()
            
            # Update job status
            job_status[job_id] = {
                "status": "completed", 
                "progress": 100, 
                "message": "Subtitles generated successfully",
                "data": {
                    "project_id": project_id,
                    "subtitles_path": subtitles_path
                }
            }
        except Exception as e:
            job_status[job_id] = {"status": "error", "message": str(e)}
    
    # Start background processing
    thread = threading.Thread(target=generate_audio_subtitles)
    thread.daemon = True
    thread.start()
    
    return jsonify({"job_id": job_id})

@app.route('/api/project/<project_id>/generate_scenes', methods=['POST'])
def generate_scenes(project_id):
    """Generate scene descriptions from subtitles"""
    # Check if required models are installed
    if not model_manager.check_model_installed('gemini'):
        return jsonify({"error": "Gemini API model not installed"}), 400
    
    # Get project subtitle file
    conn = sqlite3.connect('horror_app.db')
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT file_path FROM generated_content WHERE project_id = ? AND content_type = 'subtitles'", 
        (project_id,)
    )
    result = cursor.fetchone()
    
    if not result:
        conn.close()
        return jsonify({"error": "Subtitles content not found"}), 404
    
    subtitles_path = result[0]
    
    # Track job status
    job_id = str(time.time())
    job_status[job_id] = {"status": "running", "progress": 0, "message": "Generating scene descriptions..."}
    
    # Function to run in background thread
    def generate_scene_descriptions():
        try:
            # Initialize scene generator
            scene_gen = SceneGenerator()
            
            # Generate scene descriptions
            scenes_path = os.path.join(app.config['UPLOAD_FOLDER'], f"scenes_{project_id}.json")
            scenes = scene_gen.generate_scene_descriptions(subtitles_path)
            
            # Save scenes to file
            with open(scenes_path, 'w', encoding='utf-8') as f:
                json.dump(scenes, f, indent=2)
            
            # Save to database
            conn = sqlite3.connect('horror_app.db')
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO generated_content (project_id, content_type, file_path, date_created) VALUES (?, ?, ?, datetime('now'))",
                (project_id, "scenes", scenes_path)
            )
            
            # Update project status
            cursor.execute(
                "UPDATE projects SET status = ? WHERE id = ?",
                ("scenes_generated", project_id)
            )
            
            conn.commit()
            conn.close()
            
            # Update job status
            job_status[job_id] = {
                "status": "completed", 
                "progress": 100, 
                "message": "Scene descriptions generated successfully",
                "data": {
                    "project_id": project_id,
                    "scenes_path": scenes_path
                }
            }
        except Exception as e:
            job_status[job_id] = {"status": "error", "message": str(e)}
    
    # Start background processing
    thread = threading.Thread(target=generate_scene_descriptions)
    thread.daemon = True
    thread.start()
    
    return jsonify({"job_id": job_id})

@app.route('/api/project/<project_id>/generate_images', methods=['POST'])
def generate_images(project_id):
    """Generate images from scene descriptions"""
    # Check if required models are installed
    if not model_manager.check_model_installed('stable_diffusion'):
        return jsonify({"error": "Stable Diffusion model not installed"}), 400
    
    # Get project scene descriptions
    conn = sqlite3.connect('horror_app.db')
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT file_path FROM generated_content WHERE project_id = ? AND content_type = 'scenes'", 
        (project_id,)
    )
    result = cursor.fetchone()
    
    if not result:
        conn.close()
        return jsonify({"error": "Scene descriptions not found"}), 404
    
    scenes_path = result[0]
    
    # Get image generation preferences
    preferences = request.json
    
    # Track job status
    job_id = str(time.time())
    job_status[job_id] = {"status": "running", "progress": 0, "message": "Generating images..."}
    
    # Function to run in background thread
    def generate_scene_images():
        try:
            # Load scene descriptions
            with open(scenes_path, 'r', encoding='utf-8') as f:
                scenes = json.load(f)
            
            # Initialize image generator
            image_gen = ImageGenerator(
                style=preferences.get('style', 'cinematic')
            )
            
            # Create images directory
            images_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"images_{project_id}")
            os.makedirs(images_dir, exist_ok=True)
            
            # Generate images with progress updates
            total_scenes = len(scenes)
            image_paths = []
            
            for i, scene in enumerate(scenes):
                # Generate image
                image_path = os.path.join(images_dir, f"scene_{i+1:03d}.png")
                image_gen.generate_image(scene['description'], image_path)
                image_paths.append(image_path)
                
                # Update progress
                progress = int(((i + 1) / total_scenes) * 100)
                job_status[job_id] = {
                    "status": "running", 
                    "progress": progress, 
                    "message": f"Generated image {i+1}/{total_scenes}"
                }
            
            # Save image metadata to file
            metadata_path = os.path.join(app.config['UPLOAD_FOLDER'], f"image_metadata_{project_id}.json")
            metadata = {
                "image_count": len(image_paths),
                "images": [
                    {
                        "path": path,
                        "scene_index": i,
                        "timing": scenes[i]['start_time'] + " - " + scenes[i]['end_time']
                    } 
                    for i, path in enumerate(image_paths)
                ]
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            # Save to database
            conn = sqlite3.connect('horror_app.db')
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO generated_content (project_id, content_type, file_path, date_created) VALUES (?, ?, ?, datetime('now'))",
                (project_id, "images", metadata_path)
            )
            
            # Update project status
            cursor.execute(
                "UPDATE projects SET status = ? WHERE id = ?",
                ("images_generated", project_id)
            )
            
            conn.commit()
            conn.close()
            
            # Update job status
            job_status[job_id] = {
                "status": "completed", 
                "progress": 100, 
                "message": "Images generated successfully",
                "data": {
                    "project_id": project_id,
                    "image_count": len(image_paths),
                    "metadata_path": metadata_path
                }
            }
        except Exception as e:
            job_status[job_id] = {"status": "error", "message": str(e)}
    
    # Start background processing
    thread = threading.Thread(target=generate_scene_images)
    thread.daemon = True
    thread.start()
    
    return jsonify({"job_id": job_id})

@app.route('/api/project/<project_id>/generate_ambient', methods=['POST'])
def generate_ambient(project_id):
    """Generate ambient soundscape for a project"""
    # Check if required models are installed
    if not model_manager.check_model_installed('sound_effects'):
        return jsonify({"error": "Sound effects library not installed"}), 400
    
    # Get project scene and audio files
    conn = sqlite3.connect('horror_app.db')
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT file_path FROM generated_content WHERE project_id = ? AND content_type = 'scenes'", 
        (project_id,)
    )
    scenes_result = cursor.fetchone()
    
    cursor.execute(
        "SELECT file_path FROM generated_content WHERE project_id = ? AND content_type = 'audio'", 
        (project_id,)
    )
    audio_result = cursor.fetchone()
    
    if not scenes_result or not audio_result:
        conn.close()
        return jsonify({"error": "Scene descriptions or audio not found"}), 404
    
    scenes_path = scenes_result[0]
    audio_path = audio_result[0]
    
    # Track job status
    job_id = str(time.time())
    job_status[job_id] = {"status": "running", "progress": 0, "message": "Generating ambient soundscape..."}
    
    # Function to run in background thread
    def generate_sound_design():
        try:
            # Load scene descriptions
            with open(scenes_path, 'r', encoding='utf-8') as f:
                scenes = json.load(f)
            
            # Initialize sound designer
            sound_designer = SoundDesigner()
            
            # Generate ambient soundscape
            ambient_path = os.path.join(app.config['UPLOAD_FOLDER'], f"ambient_{project_id}.wav")
            
            # Get audio duration
            from pydub import AudioSegment
            audio_duration = len(AudioSegment.from_file(audio_path)) / 1000  # Convert ms to seconds
            
            sound_designer.generate_ambient_soundscape(scenes, audio_duration, ambient_path)
            
            # Save to database
            conn = sqlite3.connect('horror_app.db')
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO generated_content (project_id, content_type, file_path, date_created) VALUES (?, ?, ?, datetime('now'))",
                (project_id, "ambient", ambient_path)
            )
            
            # Update project status
            cursor.execute(
                "UPDATE projects SET status = ? WHERE id = ?",
                ("ambient_generated", project_id)
            )
            
            conn.commit()
            conn.close()
            
            # Update job status
            job_status[job_id] = {
                "status": "completed", 
                "progress": 100, 
                "message": "Ambient soundscape generated successfully",
                "data": {
                    "project_id": project_id,
                    "ambient_path": ambient_path
                }
            }
        except Exception as e:
            job_status[job_id] = {"status": "error", "message": str(e)}
    
    # Start background processing
    thread = threading.Thread(target=generate_sound_design)
    thread.daemon = True
    thread.start()
    
    return jsonify({"job_id": job_id})

@app.route('/api/project/<project_id>/compile_video', methods=['POST'])
def compile_video(project_id):
    """Compile the final video for a project"""
    # Get all project files
    conn = sqlite3.connect('horror_app.db')
    cursor = conn.cursor()
    
    # Get project title
    cursor.execute("SELECT title FROM projects WHERE id = ?", (project_id,))
    project_result = cursor.fetchone()
    
    if not project_result:
        conn.close()
        return jsonify({"error": "Project not found"}), 404
    
    project_title = project_result[0]
    
    # Get all content files
    cursor.execute(
        "SELECT content_type, file_path FROM generated_content WHERE project_id = ?", 
        (project_id,)
    )
    content_files = {row[0]: row[1] for row in cursor.fetchall()}
    
    # Check if we have all required content
    required_content = ['audio', 'subtitles', 'images', 'ambient']
    missing_content = [content for content in required_content if content not in content_files]
    
    if missing_content:
        conn.close()
        return jsonify({
            "error": f"Missing required content: {', '.join(missing_content)}"
        }), 400
    
    # Get video compilation preferences
    preferences = request.json
    
    # Track job status
    job_id = str(time.time())
    job_status[job_id] = {"status": "running", "progress": 0, "message": "Compiling video..."}
    
    # Function to run in background thread
    def compile_final_video():
        try:
            # Get the image metadata
            with open(content_files['images'], 'r', encoding='utf-8') as f:
                image_metadata = json.load(f)
            
            # Load scene descriptions to get timing information
            with open(content_files.get('scenes', ''), 'r', encoding='utf-8') as f:
                scenes = json.load(f)
            
            # Create image prompts in the format expected by the video compiler
            image_prompts = []
            for i, scene in enumerate(scenes):
                if i < len(image_metadata['images']):
                    image_prompts.append({
                        'timing': (scene['start_time'], scene['end_time']),
                        'prompt': scene['description'],  # Use description as prompt
                        'original_description': scene['description']
                    })
            
            # Get the image paths
            image_paths = [item['path'] for item in image_metadata['images']]
            
            # Initialize video compiler
            video_compiler = VideoCompiler(
                video_quality=preferences.get('video_quality', '4000k'),
                cinematic_ratio=preferences.get('cinematic_ratio', 2.35),
                use_dust_overlay=preferences.get('use_dust_overlay', True)
            )
            
            # Compile video
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"video_{project_id}.mp4")
            safe_title = secure_filename(project_title) if project_title else f"horror_story_{project_id}"
            
            video_compiler.create_final_video(
                image_prompts=image_prompts,
                image_paths=image_paths,
                audio_path=content_files['audio'],
                title=safe_title,
                srt_path=content_files['subtitles'],
                ambient_path=content_files['ambient']
            )
            
            # Save to database
            conn = sqlite3.connect('horror_app.db')
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO generated_content (project_id, content_type, file_path, date_created) VALUES (?, ?, ?, datetime('now'))",
                (project_id, "video", output_path)
            )
            
            # Update project status
            cursor.execute(
                "UPDATE projects SET status = ? WHERE id = ?",
                ("completed", project_id)
            )
            
            conn.commit()
            conn.close()
            
            # Update job status
            job_status[job_id] = {
                "status": "completed", 
                "progress": 100, 
                "message": "Video compiled successfully",
                "data": {
                    "project_id": project_id,
                    "video_path": output_path
                }
            }
        except Exception as e:
            job_status[job_id] = {"status": "error", "message": str(e)}
    
    # Start background processing
    thread = threading.Thread(target=compile_final_video)
    thread.daemon = True
    thread.start()
    
    return jsonify({"job_id": job_id})

@app.route('/api/projects', methods=['GET'])
def list_projects():
    """List all projects"""
    conn = sqlite3.connect('horror_app.db')
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT id, title, date_created, status FROM projects ORDER BY date_created DESC"
    )
    projects = [
        {
            "id": row[0],
            "title": row[1],
            "date_created": row[2],
            "status": row[3]
        }
        for row in cursor.fetchall()
    ]
    
    conn.close()
    
    return jsonify(projects)

@app.route('/api/project/<project_id>', methods=['GET'])
def get_project(project_id):
    """Get details of a specific project"""
    conn = sqlite3.connect('horror_app.db')
    cursor = conn.cursor()
    
    # Get project details
    cursor.execute(
        "SELECT id, title, date_created, status, settings, story_id, subreddit FROM projects WHERE id = ?",
        (project_id,)
    )
    project_row = cursor.fetchone()
    
    if not project_row:
        conn.close()
        return jsonify({"error": "Project not found"}), 404
    
    project = {
        "id": project_row[0],
        "title": project_row[1],
        "date_created": project_row[2],
        "status": project_row[3],
        "settings": json.loads(project_row[4]) if project_row[4] else None,
        "story_id": project_row[5],
        "subreddit": project_row[6]
    }
    
    # Get all generated content
    cursor.execute(
        "SELECT content_type, file_path, date_created FROM generated_content WHERE project_id = ?",
        (project_id,)
    )
    
    content = []
    for row in cursor.fetchall():
        content.append({
            "type": row[0],
            "file_path": row[1],
            "date_created": row[2]
        })
    
    project["content"] = content
    conn.close()
    
    return jsonify(project)

@app.route('/api/content/<path:file_path>', methods=['GET'])
def get_content(file_path):
    """Serve a generated content file"""
    # Security check - make sure the file is in the upload folder
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], file_path)
    if not os.path.abspath(full_path).startswith(os.path.abspath(app.config['UPLOAD_FOLDER'])):
        return jsonify({"error": "Access denied"}), 403
    
    if not os.path.exists(full_path):
        return jsonify({"error": "File not found"}), 404
    
    return send_file(full_path)

if __name__ == '__main__':
    app.run(debug=True) 