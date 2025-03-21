import os
import whisper
from whisper.utils import WriteSRT
import torch
from typing import Optional

class SubtitleGenerator:
    """Generates subtitles from audio files"""
    
    def __init__(self, model_size="base"):
        """Initialize with Whisper model size"""
        self.model_size = model_size
        
        # Check if Whisper model is installed
        self.model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'models', 'whisper'
        )
        
        if not os.path.exists(self.model_path):
            print("Warning: Whisper model directory not found. Subtitle generation may fail.")
    
    def generate_subtitles(self, audio_path: str, output_path: str) -> str:
        """Generate SRT subtitles from audio file"""
        try:
            # Make sure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Load Whisper model
            model = whisper.load_model(self.model_size)
            
            # Transcribe with timing info
            result = model.transcribe(
                audio_path,
                verbose=False,
                word_timestamps=True,
                fp16=torch.cuda.is_available()
            )
            
            # Save SRT file
            with open(output_path, "w", encoding="utf-8") as srt_file:
                writer = WriteSRT(os.path.dirname(output_path))
                writer.write_result(result, srt_file)
            
            print(f"Subtitles generated: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error generating subtitles: {str(e)}")
            raise 