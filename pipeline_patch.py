# This file contains patches to apply to pipeline.py to fix the user_prefs issue

# Define HORROR_SUBREDDITS at the top level
HORROR_SUBREDDITS = ["nosleep", "shortscarystories", "creepypasta", "LetsNotMeet", 
                     "DarkTales", "TheCrypticCompendium", "libraryofshadows", 
                     "scarystories", "TrueScaryStories", "HorrorStories"]

# Create a default user_prefs dictionary
user_prefs = {
    'subreddits': {'value': ["nosleep", "shortscarystories"]},
    'min_length': {'value': 1000},
    'voice_speed': {'value': 0.85},
    'voice_selection': {'value': 'af_bella'},
    'video_quality': {'value': '4000k'},
    'cinematic_ratio': {'value': 2.35},
    'use_dust_overlay': {'value': True}
}

# This function will be used to apply the patch
def apply_pipeline_patch():
    """Apply patches to the pipeline.py file to fix the user_prefs issue"""
    import pipeline
    
    # Set the HORROR_SUBREDDITS variable
    pipeline.HORROR_SUBREDDITS = HORROR_SUBREDDITS
    
    # Set the user_prefs variable
    pipeline.user_prefs = user_prefs
    
    print("Pipeline patches applied successfully") 