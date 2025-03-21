import os
import re
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import random
from typing import Dict, List, Optional

class SoundDesigner:
    """Creates contextual ambient sound design for horror stories"""
    
    def __init__(self, sound_library_path=None):
        """Initialize with path to sound effect library"""
        if sound_library_path is None:
            sound_library_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                'models', 'sound_effects'
            )
        
        self.sound_library_path = sound_library_path
        self.sound_categories = {
            'indoor': ['creaking', 'footsteps', 'door', 'clock', 'breathing', 'whisper'],
            'outdoor': ['wind', 'rain', 'thunder', 'leaves', 'branches', 'animals'],
            'tension': ['heartbeat', 'drone', 'strings', 'pulse', 'rumble', 'static'],
            'horror': ['scream', 'whisper', 'laugh', 'growl', 'scratch', 'thump']
        }
        
        # Create sound library directory if it doesn't exist
        os.makedirs(self.sound_library_path, exist_ok=True)
        
        # Check if sound library is empty
        if not os.listdir(self.sound_library_path):
            print("Warning: Sound effects library is empty. Ambient sound generation may fail.")
            self._create_placeholder_sounds()
    
    def _create_placeholder_sounds(self):
        """Create placeholder sound files for testing"""
        for category in self.sound_categories:
            for sound_type in self.sound_categories[category][:2]:  # Just create a couple for each category
                # Create a placeholder audio file
                sample_rate = 44100
                duration = 5  # 5 seconds
                t = np.linspace(0, duration, int(sample_rate * duration))
                
                # Create different tones for different categories
                if category == 'indoor':
                    freq = 220
                elif category == 'outdoor':
                    freq = 440
                elif category == 'tension':
                    freq = 110
                else:  # horror
                    freq = 880
                
                # Generate a simple sine wave
                audio = np.sin(2 * np.pi * freq * t) * 0.1
                
                # Convert to int16 format
                audio = (audio * 32767).astype(np.int16)
                
                # Create an AudioSegment
                audio_segment = AudioSegment(
                    audio.tobytes(),
                    frame_rate=sample_rate,
                    sample_width=2,  # 16-bit audio
                    channels=1  # Mono
                )
                
                # Save the audio file
                output_path = os.path.join(self.sound_library_path, f"{category}_{sound_type}.wav")
                audio_segment.export(output_path, format="wav")
                
                print(f"Created placeholder sound: {output_path}")
    
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
    
    def convert_timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert SRT timestamp format to seconds"""
        parts = timestamp.replace(',', '.').split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    
    def generate_ambient_soundscape(self, scene_descriptions: List[Dict], audio_duration: float, output_path: str) -> str:
        """Create a complete ambient sound mix for the entire story"""
        # Create base silent track
        base_track = AudioSegment.silent(duration=int(audio_duration * 1000))
        
        # Process each scene
        for i, scene in enumerate(scene_descriptions):
            # Calculate scene timing
            start_time = self.convert_timestamp_to_seconds(scene['start_time'])
            end_time = self.convert_timestamp_to_seconds(scene['end_time'])
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