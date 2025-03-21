import os
import json
import time
import requests
import threading
import hashlib
import shutil
import zipfile
from typing import Dict, List, Optional

class ModelManager:
    """Manages AI models, their installation status, and downloading"""
    
    def __init__(self, models_dir: str):
        """Initialize the model manager"""
        self.models_dir = models_dir
        self.model_info_file = os.path.join(models_dir, "model_info.json")
        self.download_status = {}
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize model information
        self._init_model_info()
    
    def _init_model_info(self):
        """Initialize or load model information"""
        if os.path.exists(self.model_info_file):
            with open(self.model_info_file, 'r') as f:
                self.model_info = json.load(f)
        else:
            self.model_info = self._get_default_model_info()
            self._save_model_info()
        
        # Check current installation status
        self._update_installation_status()
    
    def _get_default_model_info(self) -> Dict:
        """Define default model information"""
        return {
            "models": {
                "gemini": {
                    "name": "Google Gemini API",
                    "description": "AI language model for story enhancement and image prompts",
                    "version": "2.0-flash",
                    "size_mb": 0,  # API-based, no download required
                    "download_url": None,
                    "installation_folder": "gemini",
                    "installed": False,
                    "api_key_required": True
                },
                "whisper": {
                    "name": "Whisper ASR",
                    "description": "Automatic speech recognition for subtitle generation",
                    "version": "base",
                    "size_mb": 142,
                    "download_url": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879326a3c35dfe8af8865a29e8506cc8a9c8ff6981047e0ded/base.pt",
                    "installation_folder": "whisper",
                    "installed": False,
                    "api_key_required": False
                },
                "tts": {
                    "name": "Kokoro TTS",
                    "description": "Text-to-speech model for voice generation",
                    "version": "1.0",
                    "size_mb": 350,
                    "download_url": "https://example.com/kokoro_tts.zip",  # Placeholder URL
                    "installation_folder": "tts",
                    "installed": False,
                    "api_key_required": False
                },
                "stable_diffusion": {
                    "name": "Stable Diffusion XL",
                    "description": "Image generation model for cinematic visuals",
                    "version": "1.0",
                    "size_mb": 6800,
                    "download_url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
                    "installation_folder": "stable_diffusion",
                    "installed": False,
                    "api_key_required": False
                },
                "sound_effects": {
                    "name": "Horror Sound Effects Library",
                    "description": "Sound effects for ambient sound generation",
                    "version": "1.0",
                    "size_mb": 250,
                    "download_url": "https://example.com/horror_sounds.zip",  # Placeholder URL
                    "installation_folder": "sound_effects",
                    "installed": False,
                    "api_key_required": False
                }
            }
        }
    
    def _save_model_info(self):
        """Save model information to file"""
        with open(self.model_info_file, 'w') as f:
            json.dump(self.model_info, f, indent=2)
    
    def _update_installation_status(self):
        """Update installation status for all models based on file presence"""
        for model_id, model in self.model_info["models"].items():
            model_path = os.path.join(self.models_dir, model["installation_folder"])
            
            # API-based models (like Gemini) don't need file checks
            if model["download_url"] is None and model["api_key_required"]:
                # For API models, check if API key is set
                api_key_file = os.path.join(model_path, "api_key.txt")
                model["installed"] = os.path.exists(api_key_file)
            else:
                # For file-based models, check if model directory exists and has files
                model["installed"] = (
                    os.path.exists(model_path) and 
                    os.path.isdir(model_path) and 
                    len(os.listdir(model_path)) > 0
                )
                
        self._save_model_info()
    
    def list_models(self) -> Dict:
        """List all available models and their status"""
        self._update_installation_status()
        return self.model_info
    
    def get_available_models(self) -> List[str]:
        """Get a list of available model IDs"""
        return list(self.model_info["models"].keys())
    
    def check_model_installed(self, model_id: str) -> bool:
        """Check if a model is installed"""
        if model_id not in self.model_info["models"]:
            return False
        
        self._update_installation_status()
        return self.model_info["models"][model_id]["installed"]
    
    def get_model_status(self, model_id: str) -> Dict:
        """Get the current status of a model"""
        if model_id not in self.model_info["models"]:
            return {"error": "Model not found"}
        
        model = self.model_info["models"][model_id]
        status = {
            "id": model_id,
            "name": model["name"],
            "installed": model["installed"],
        }
        
        # Add download progress if available
        if model_id in self.download_status:
            status.update(self.download_status[model_id])
        
        return status
    
    def save_api_key(self, model_id: str, api_key: str) -> bool:
        """Save API key for an API-based model"""
        if model_id not in self.model_info["models"] or not self.model_info["models"][model_id]["api_key_required"]:
            return False
        
        model_path = os.path.join(self.models_dir, self.model_info["models"][model_id]["installation_folder"])
        os.makedirs(model_path, exist_ok=True)
        
        api_key_file = os.path.join(model_path, "api_key.txt")
        with open(api_key_file, 'w') as f:
            f.write(api_key)
        
        self._update_installation_status()
        return True
    
    def get_api_key(self, model_id: str) -> Optional[str]:
        """Get API key for an API-based model"""
        if model_id not in self.model_info["models"] or not self.model_info["models"][model_id]["api_key_required"]:
            return None
        
        model_path = os.path.join(self.models_dir, self.model_info["models"][model_id]["installation_folder"])
        api_key_file = os.path.join(model_path, "api_key.txt")
        
        if not os.path.exists(api_key_file):
            return None
        
        with open(api_key_file, 'r') as f:
            return f.read().strip()
    
    def download_model(self, model_id: str):
        """Download a model"""
        if model_id not in self.model_info["models"]:
            self.download_status[model_id] = {
                "status": "error",
                "message": "Model not found"
            }
            return
        
        model = self.model_info["models"][model_id]
        
        # Skip API-based models
        if model["download_url"] is None:
            self.download_status[model_id] = {
                "status": "error",
                "message": "This model requires an API key instead of download"
            }
            return
        
        # Create model directory
        model_path = os.path.join(self.models_dir, model["installation_folder"])
        os.makedirs(model_path, exist_ok=True)
        
        # Initialize download status
        self.download_status[model_id] = {
            "status": "downloading",
            "progress": 0,
            "message": "Starting download..."
        }
        
        try:
            # Download the model
            download_url = model["download_url"]
            local_filename = os.path.join(model_path, os.path.basename(download_url))
            
            # Stream download with progress updates
            response = requests.get(download_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(local_filename, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress = int(100 * downloaded / total_size) if total_size > 0 else 0
                        
                        self.download_status[model_id] = {
                            "status": "downloading",
                            "progress": progress,
                            "message": f"Downloading... {progress}%"
                        }
            
            # Extract if it's a zip file
            if download_url.endswith('.zip'):
                self.download_status[model_id] = {
                    "status": "extracting",
                    "progress": 0,
                    "message": "Extracting files..."
                }
                
                with zipfile.ZipFile(local_filename, 'r') as zip_ref:
                    zip_ref.extractall(model_path)
                
                # Remove the zip file after extraction
                os.remove(local_filename)
            
            # Update status
            self.download_status[model_id] = {
                "status": "completed",
                "progress": 100,
                "message": "Download completed!"
            }
            
            # Update installation status
            self._update_installation_status()
            
        except Exception as e:
            self.download_status[model_id] = {
                "status": "error",
                "message": f"Download failed: {str(e)}"
            } 