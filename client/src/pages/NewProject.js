import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './NewProject.css';

function NewProject() {
  const navigate = useNavigate();
  const [subreddits, setSubreddits] = useState([]);
  const [availableSubreddits, setAvailableSubreddits] = useState([]);
  const [preferences, setPreferences] = useState({
    subreddits: ['nosleep', 'shortscarystories'],
    min_length: 1000,
    voice_speed: 0.85,
    voice_selection: 'en_emily',
    video_quality: '4000k',
    cinematic_ratio: 2.35,
    use_dust_overlay: true
  });
  const [modelStatus, setModelStatus] = useState({});
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentStep, setCurrentStep] = useState(null);
  const [error, setError] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [projectId, setProjectId] = useState(null);

  useEffect(() => {
    // Fetch available subreddits
    fetch('/api/reddit/horror_subreddits')
      .then(res => res.json())
      .then(data => setAvailableSubreddits(data))
      .catch(err => console.error('Error fetching subreddits:', err));
    
    // Check model status
    fetch('/api/models')
      .then(res => res.json())
      .then(data => {
        const status = {
          gemini: data.models.gemini.installed,
          whisper: data.models.whisper.installed,
          tts: data.models.tts.installed,
          stable_diffusion: data.models.stable_diffusion.installed,
          sound_effects: data.models.sound_effects.installed
        };
        setModelStatus(status);
      })
      .catch(err => console.error('Error fetching model status:', err));
  }, []);

  // Poll job status when we have a job running
  useEffect(() => {
    if (!jobId) return;
    
    const intervalId = setInterval(() => {
      checkJobStatus(jobId);
    }, 2000);
    
    return () => clearInterval(intervalId);
  }, [jobId]);

  const handlePreferenceChange = (key, value) => {
    setPreferences({
      ...preferences,
      [key]: value
    });
  };

  const handleSubredditChange = (subreddit, isChecked) => {
    let updatedSubreddits;
    if (isChecked) {
      updatedSubreddits = [...preferences.subreddits, subreddit];
    } else {
      updatedSubreddits = preferences.subreddits.filter(sr => sr !== subreddit);
    }
    
    setPreferences({
      ...preferences,
      subreddits: updatedSubreddits
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validate that we have at least one subreddit selected
    if (preferences.subreddits.length === 0) {
      setError('Please select at least one subreddit');
      return;
    }
    
    // Check if all required models are installed
    const requiredModels = ['gemini', 'whisper', 'tts', 'stable_diffusion', 'sound_effects'];
    const missingModels = requiredModels.filter(modelId => !modelStatus[modelId]);
    
    if (missingModels.length > 0) {
      setError(`The following required models are not installed: ${missingModels.join(', ')}. Please install them in the Models section.`);
      return;
    }
    
    // Start the generation process
    setIsGenerating(true);
    setCurrentStep('starting');
    setError(null);
    
    try {
      const response = await fetch('/api/projects', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          preferences: preferences
        })
      });
      
      const data = await response.json();
      
      if (response.ok) {
        setJobId(data.job_id);
        setProjectId(data.project_id);
        setCurrentStep('fetching_story');
      } else {
        throw new Error(data.error || 'Failed to start generation');
      }
    } catch (err) {
      setError(`Failed to start generation: ${err.message}`);
      setIsGenerating(false);
      setCurrentStep(null);
    }
  };

  const checkJobStatus = async (jobId) => {
    try {
      const response = await fetch(`/api/jobs/${jobId}`);
      const data = await response.json();
      
      if (response.ok) {
        setCurrentStep(data.current_step);
        
        // If complete, redirect to the project page
        if (data.status === 'completed') {
          navigate(`/projects/${projectId}`);
        }
        
        // If failed, show error
        if (data.status === 'failed') {
          setError(`Generation failed: ${data.error || 'Unknown error'}`);
          setIsGenerating(false);
          setJobId(null);
        }
      } else {
        throw new Error(data.error || 'Failed to check job status');
      }
    } catch (err) {
      console.error('Error checking job status:', err);
    }
  };

  const renderProgressStep = () => {
    const steps = [
      { id: 'starting', label: 'Starting generation process...' },
      { id: 'fetching_story', label: 'Fetching horror story from Reddit...' },
      { id: 'enhancing_story', label: 'Enhancing story for narration...' },
      { id: 'generating_voice', label: 'Generating voice-over narration...' },
      { id: 'creating_subtitles', label: 'Creating subtitles from audio...' },
      { id: 'generating_scenes', label: 'Generating cinematic scene descriptions...' },
      { id: 'creating_images', label: 'Creating cinematic images...' },
      { id: 'designing_ambient', label: 'Designing ambient soundscape...' },
      { id: 'compiling_video', label: 'Compiling final video...' },
      { id: 'completed', label: 'Generation complete!' },
    ];
    
    return (
      <div className="progress-steps">
        <h2>Creating Your Horror Narration</h2>
        {steps.map(step => (
          <div 
            key={step.id} 
            className={`progress-step ${currentStep === step.id ? 'active' : ''} ${
              steps.findIndex(s => s.id === step.id) < steps.findIndex(s => s.id === currentStep) ? 'completed' : ''
            }`}
          >
            <div className="step-indicator"></div>
            <div className="step-label">{step.label}</div>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="new-project">
      <h1>Create New Horror Story Video</h1>
      
      {error && (
        <div className="error-message">
          <p>{error}</p>
          <button onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}
      
      {isGenerating ? (
        renderProgressStep()
      ) : (
        <form onSubmit={handleSubmit}>
          <div className="form-section">
            <h2>Story Sources</h2>
            <p>Select subreddits to fetch horror stories from:</p>
            <div className="subreddit-grid">
              {availableSubreddits.map(subreddit => (
                <div className="subreddit-option" key={subreddit}>
                  <input
                    type="checkbox"
                    id={`subreddit-${subreddit}`}
                    checked={preferences.subreddits.includes(subreddit)}
                    onChange={(e) => handleSubredditChange(subreddit, e.target.checked)}
                  />
                  <label htmlFor={`subreddit-${subreddit}`}>r/{subreddit}</label>
                </div>
              ))}
            </div>
          </div>
          
          <div className="form-section">
            <h2>Voice Settings</h2>
            <div className="form-group">
              <label htmlFor="voice-selection">Narrator Voice:</label>
              <select
                id="voice-selection"
                value={preferences.voice_selection}
                onChange={(e) => handlePreferenceChange('voice_selection', e.target.value)}
              >
                <option value="en_emily">Emily (Female)</option>
                <option value="en_ryan">Ryan (Male)</option>
                <option value="en_ghost">Ghost (Ethereal)</option>
                <option value="en_whisper">Whisper (Creepy)</option>
              </select>
            </div>
            
            <div className="form-group">
              <label htmlFor="voice-speed">Narration Speed: {preferences.voice_speed}x</label>
              <input
                type="range"
                id="voice-speed"
                min="0.7"
                max="1.0"
                step="0.05"
                value={preferences.voice_speed}
                onChange={(e) => handlePreferenceChange('voice_speed', parseFloat(e.target.value))}
              />
            </div>
          </div>
          
          <div className="form-section">
            <h2>Video Settings</h2>
            <div className="form-group">
              <label htmlFor="video-quality">Video Quality:</label>
              <select
                id="video-quality"
                value={preferences.video_quality}
                onChange={(e) => handlePreferenceChange('video_quality', e.target.value)}
              >
                <option value="2000k">Low (2000k)</option>
                <option value="4000k">Medium (4000k)</option>
                <option value="8000k">High (8000k)</option>
              </select>
            </div>
            
            <div className="form-group">
              <label htmlFor="cinematic-ratio">Cinematic Ratio:</label>
              <select
                id="cinematic-ratio"
                value={preferences.cinematic_ratio}
                onChange={(e) => handlePreferenceChange('cinematic_ratio', parseFloat(e.target.value))}
              >
                <option value="2.35">2.35:1 (Cinematic)</option>
                <option value="1.85">1.85:1 (Widescreen)</option>
                <option value="1.78">16:9 (Standard)</option>
              </select>
            </div>
            
            <div className="form-group checkbox-group">
              <input
                type="checkbox"
                id="dust-overlay"
                checked={preferences.use_dust_overlay}
                onChange={(e) => handlePreferenceChange('use_dust_overlay', e.target.checked)}
              />
              <label htmlFor="dust-overlay">Add cinematic dust overlay for atmosphere</label>
            </div>
          </div>
          
          <div className="form-actions">
            <button type="submit" className="primary-button">
              Generate Horror Video
            </button>
          </div>
        </form>
      )}
    </div>
  );
}

export default NewProject; 