import os
import re
import json
from google import genai
import time
from typing import Dict, List, Optional

class SceneGenerator:
    """Generates cinematic scene descriptions from subtitles"""
    
    def __init__(self, delay_seconds=2):
        """Initialize with API settings"""
        self.delay_seconds = delay_seconds
        
        # Initialize Gemini API client
        api_key = self._get_gemini_api_key()
        if api_key:
            self.gemini_client = genai.Client(api_key=api_key)
        else:
            self.gemini_client = None
            print("Warning: Gemini API key not found. Scene generation disabled.")
    
    def _get_gemini_api_key(self) -> Optional[str]:
        """Get Gemini API key from models directory"""
        api_key_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'models', 'gemini', 'api_key.txt'
        )
        
        if os.path.exists(api_key_file):
            with open(api_key_file, 'r') as f:
                return f.read().strip()
        return None
    
    def parse_srt_timestamps(self, srt_path: str) -> List[Dict]:
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
    
    def generate_scene_descriptions(self, srt_path: str) -> List[Dict]:
        """Generate cinematic scene descriptions based on subtitle segments"""
        # Check if Gemini API is available
        if not self.gemini_client:
            raise ValueError("Gemini API client not initialized. Cannot generate scene descriptions.")
        
        segments = self.parse_srt_timestamps(srt_path)
        scene_descriptions = []
        
        chunk_size = 2
        max_retries = 3
        
        for i in range(0, len(segments) - chunk_size + 1, chunk_size):
            chunk = segments[i:i + chunk_size]
            combined_text = ' '.join([seg['text'] for seg in chunk])
            
            # Enhanced prompt focusing on visual storytelling
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
                    response = self.gemini_client.models.generate_content(
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
                    time.sleep(self.delay_seconds)
                    break
                    
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:  # Rate limit error
                        wait_time = (attempt + 1) * self.delay_seconds * 2
                        print(f"Rate limit hit, waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        print(f"Error generating scene: {str(e)}")
                        fallback_desc = "A dimly lit room with shadows stretching across the walls. A figure stands motionless, their face obscured by darkness as moonlight filters through a nearby window."
                        scene_descriptions.append({
                            'start_time': chunk[0]['start_time'],
                            'end_time': chunk[-1]['end_time'],
                            'description': fallback_desc
                        })
                        break
        
        return scene_descriptions 