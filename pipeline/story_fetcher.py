import praw
import random
import os
import re
from google import genai
from typing import Dict, List, Optional

class StoryFetcher:
    """Fetches and enhances horror stories from Reddit"""
    
    def __init__(self, subreddits=None, min_length=1000):
        """Initialize with subreddit preferences"""
        self.subreddits = subreddits or ["nosleep", "shortscarystories"]
        self.min_length = min_length
        self.horror_themes = [
            "paranormal", "ghost", "haunting", "demon", "possession",
            "monster", "creature", "stalker", "serial killer", "unexplained",
            "ritual", "cult", "ancient evil", "cursed", "shadow people",
            "sleep paralysis", "night terror", "abandoned", "forest", "cabin",
            "basement", "attic", "mirror", "doppelganger", "entity"
        ]
        
        # Initialize Reddit API client
        self.reddit = praw.Reddit(
            client_id="Jf3jkA3Y0dBCfluYvS8aVw",
            client_secret="1dWKIP6ME7FBR66motXS6273rkkf0g",
            user_agent="Horror Stories by Wear_Severe"
        )
        
        # Initialize Gemini API client
        api_key = self._get_gemini_api_key()
        if api_key:
            self.gemini_client = genai.Client(api_key=api_key)
        else:
            self.gemini_client = None
            print("Warning: Gemini API key not found. Story enhancement disabled.")
    
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
    
    def fetch_and_enhance_story(self) -> Dict:
        """Fetch a story from horror subreddits and enhance it"""
        # Randomly select 2-3 subreddits
        all_subreddits = [
            "nosleep", "shortscarystories", "creepypasta", "LetsNotMeet",
            "DarkTales", "TheCrypticCompendium", "libraryofshadows",
            "scarystories", "TrueScaryStories", "HorrorStories"
        ]
        selected_subreddits = self.subreddits or random.sample(all_subreddits, min(3, len(all_subreddits)))
        
        # Fetch stories from selected subreddits
        all_posts = []
        for subreddit_name in selected_subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                posts = list(subreddit.top("week", limit=30))
                all_posts.extend(posts)
            except Exception as e:
                print(f"Error fetching from r/{subreddit_name}: {str(e)}")
        
        # Shuffle posts to randomize selection
        random.shuffle(all_posts)
        
        # Filter out very short posts and check a cache of previously used posts
        cache_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "used_story_ids.txt")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        used_ids = set()
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                used_ids = set(line.strip() for line in f.readlines())
        
        filtered_posts = [
            post for post in all_posts 
            if post.id not in used_ids 
            and len(post.selftext) > self.min_length
        ]
        
        if not filtered_posts:
            filtered_posts = [post for post in all_posts if len(post.selftext) > self.min_length]
        
        # Select a story
        if not filtered_posts:
            raise ValueError("No suitable stories found. Try changing the subreddit or length criteria.")
        
        # Take a subset of posts for selection
        selection_posts = filtered_posts[:min(20, len(filtered_posts))]
        
        # Select the best story using Gemini if available
        if self.gemini_client:
            # Create a prompt for story selection
            post_titles = "\n".join([f"{i+1}. {post.title}" for i, post in enumerate(selection_posts)])
            
            selection_prompt = f"""Select ONE story number (1-{len(selection_posts)}) that has the strongest potential for a horror podcast narrative. Consider:
            - Clear narrative structure
            - Strong character development
            - Unique premise
            - Visual storytelling potential
            - Atmospheric content

            Available stories:
            {post_titles}

            Return only the number."""
            
            # Get story selection
            try:
                response = self.gemini_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=selection_prompt
                ).text
                
                story_index = int(response.strip()) - 1
                chosen_story = selection_posts[story_index]
            except (ValueError, IndexError) as e:
                chosen_story = random.choice(selection_posts)
        else:
            # If Gemini is not available, select a random story
            chosen_story = random.choice(selection_posts)
        
        # Save story ID to avoid reuse
        with open(cache_file, 'a') as f:
            f.write(f"{chosen_story.id}\n")
        
        # Enhance the story with Gemini if available
        if self.gemini_client:
            enhancement_prompt = """Transform this story into a voice over script with the following structure:

            1. Start with a powerful hook about the story's theme (2-3 sentences)
            2. Include this intro: "Welcome to The Withering Club, where we explore the darkest corners of human experience. I'm your host, Anna. Before we begin tonight's story, remember that the shadows you see might be watching back. Now, dim the lights and prepare yourself for tonight's tale..."
            3. Tell the story with a clear beginning, middle, and end, focusing on:
               - Clear narrative flow
               - Building tension
               - Natural dialogue
               - Atmospheric descriptions
            4. End with: "That concludes tonight's tale from The Withering Club. If this story kept you up at night, remember to like, share, and subscribe to join our growing community of darkness seekers. Until next time, remember... the best stories are the ones that follow you home. Sleep well, if you can."

            Original Story: {content}

            Return ONLY the complete script text with no additional formatting, explanations, or markdown."""
            
            try:
                enhanced_story = self.gemini_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=enhancement_prompt.format(content=chosen_story.selftext)
                ).text
                
                # Clean up the enhanced story
                enhanced_story = enhanced_story.strip()
            except Exception as e:
                print(f"Error enhancing story: {str(e)}")
                enhanced_story = chosen_story.selftext
        else:
            # If Gemini is not available, use the original story
            enhanced_story = chosen_story.selftext
        
        return {
            'title': chosen_story.title,
            'original': chosen_story.selftext,
            'enhanced': enhanced_story,
            'subreddit': chosen_story.subreddit.display_name,
            'story_id': chosen_story.id
        } 