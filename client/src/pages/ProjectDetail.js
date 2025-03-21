import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import './ProjectDetail.css';

function ProjectDetail() {
  const { id } = useParams();
  const [project, setProject] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('summary');

  useEffect(() => {
    fetchProject();
    
    // If the project is in processing state, poll for updates
    const intervalId = setInterval(() => {
      if (project && project.status === 'processing') {
        fetchProject();
      }
    }, 5000);
    
    return () => clearInterval(intervalId);
  }, [id, project?.status]);

  const fetchProject = async () => {
    try {
      const response = await fetch(`/api/projects/${id}`);
      const data = await response.json();
      
      if (response.ok) {
        setProject(data);
      } else {
        throw new Error(data.error || 'Failed to load project');
      }
    } catch (err) {
      setError(err.message);
      console.error('Error fetching project:', err);
    } finally {
      setLoading(false);
    }
  };

  const findContentByType = (type) => {
    if (!project || !project.content) return null;
    return project.content.find(item => item.type === type);
  };

  const renderSummary = () => {
    return (
      <div className="project-summary">
        <div className="summary-grid">
          <div className="summary-card">
            <h3>Story Details</h3>
            <p><strong>Title:</strong> {project.title}</p>
            <p><strong>Source:</strong> r/{project.subreddit}</p>
            <p><strong>Status:</strong> <span className={`status-text ${project.status}`}>{project.status}</span></p>
            <p><strong>Date Created:</strong> {new Date(project.date_created).toLocaleString()}</p>
          </div>
          
          <div className="summary-card">
            <h3>Generation Settings</h3>
            {project.settings && (
              <>
                <p><strong>Voice:</strong> {project.settings.voice_selection}</p>
                <p><strong>Voice Speed:</strong> {project.settings.voice_speed}x</p>
                <p><strong>Video Quality:</strong> {project.settings.video_quality}</p>
                <p><strong>Cinematic Ratio:</strong> {project.settings.cinematic_ratio}:1</p>
              </>
            )}
          </div>
        </div>
        
        {project.status === 'completed' && (
          <div className="final-video">
            <h3>Final Video</h3>
            <div className="video-container">
              {findContentByType('video') ? (
                <video controls>
                  <source src={`/api/content/${findContentByType('video').file_path}`} type="video/mp4" />
                  Your browser does not support the video tag.
                </video>
              ) : (
                <div className="placeholder-video">
                  <p>Video preview not available</p>
                </div>
              )}
            </div>
            <div className="video-actions">
              <a 
                href={`/api/content/${findContentByType('video')?.file_path}`} 
                download 
                className="download-button"
                disabled={!findContentByType('video')}
              >
                Download Video
              </a>
            </div>
          </div>
        )}
        
        {project.status === 'processing' && (
          <div className="processing-status">
            <h3>Processing</h3>
            <div className="processing-spinner"></div>
            <p>Your horror video is currently being generated.</p>
            <p>This might take some time depending on the story length and settings.</p>
          </div>
        )}
        
        {project.status === 'failed' && (
          <div className="failed-status">
            <h3>Generation Failed</h3>
            <p>Unfortunately, there was an error generating your horror video.</p>
            <Link to="/new-project" className="retry-button">Try Again</Link>
          </div>
        )}
      </div>
    );
  };

  const renderStory = () => {
    const storyContent = findContentByType('story');
    
    return (
      <div className="story-content">
        <h3>Horror Story</h3>
        {storyContent ? (
          <div className="story-text">
            {storyContent.content || "Story content not available"}
          </div>
        ) : (
          <div className="content-placeholder">
            <p>Story content not available yet.</p>
          </div>
        )}
      </div>
    );
  };

  const renderImages = () => {
    const images = project.content?.filter(item => item.type === 'image') || [];
    
    return (
      <div className="images-content">
        <h3>Generated Images ({images.length})</h3>
        {images.length > 0 ? (
          <div className="image-grid">
            {images.map((image, index) => (
              <div className="image-card" key={index}>
                <img src={`/api/content/${image.file_path}`} alt={`Scene ${index + 1}`} />
                <div className="image-number">Scene {index + 1}</div>
              </div>
            ))}
          </div>
        ) : (
          <div className="content-placeholder">
            <p>No images have been generated yet.</p>
          </div>
        )}
      </div>
    );
  };

  const renderAudio = () => {
    const voiceOver = findContentByType('voice_over');
    const ambientSound = findContentByType('ambient_sound');
    
    return (
      <div className="audio-content">
        <h3>Audio Files</h3>
        
        <div className="audio-section">
          <h4>Voice-over Narration</h4>
          {voiceOver ? (
            <div className="audio-player">
              <audio controls>
                <source src={`/api/content/${voiceOver.file_path}`} type="audio/wav" />
                Your browser does not support the audio element.
              </audio>
              <a 
                href={`/api/content/${voiceOver.file_path}`} 
                download 
                className="download-link"
              >
                Download Voice-over
              </a>
            </div>
          ) : (
            <div className="content-placeholder">
              <p>Voice-over not available yet.</p>
            </div>
          )}
        </div>
        
        <div className="audio-section">
          <h4>Ambient Sound Design</h4>
          {ambientSound ? (
            <div className="audio-player">
              <audio controls>
                <source src={`/api/content/${ambientSound.file_path}`} type="audio/wav" />
                Your browser does not support the audio element.
              </audio>
              <a 
                href={`/api/content/${ambientSound.file_path}`} 
                download 
                className="download-link"
              >
                Download Ambient Sound
              </a>
            </div>
          ) : (
            <div className="content-placeholder">
              <p>Ambient sound not available yet.</p>
            </div>
          )}
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="project-detail">
        <h1>Project Details</h1>
        <div className="loading">Loading project details...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="project-detail">
        <h1>Project Details</h1>
        <div className="error-message">
          <p>{error}</p>
          <Link to="/projects" className="back-button">Back to Projects</Link>
        </div>
      </div>
    );
  }

  if (!project) {
    return (
      <div className="project-detail">
        <h1>Project Not Found</h1>
        <p>The requested project could not be found.</p>
        <Link to="/projects" className="back-button">Back to Projects</Link>
      </div>
    );
  }

  return (
    <div className="project-detail">
      <div className="detail-header">
        <div className="title-section">
          <h1>{project.title}</h1>
          <span className={`status-badge ${project.status}`}>
            {project.status === 'completed' ? 'Completed' :
             project.status === 'processing' ? 'Processing' :
             project.status === 'failed' ? 'Failed' : 'Draft'}
          </span>
        </div>
        <Link to="/projects" className="back-button">Back to Projects</Link>
      </div>
      
      <div className="detail-tabs">
        <button 
          className={`tab-button ${activeTab === 'summary' ? 'active' : ''}`}
          onClick={() => setActiveTab('summary')}
        >
          Summary
        </button>
        <button 
          className={`tab-button ${activeTab === 'story' ? 'active' : ''}`}
          onClick={() => setActiveTab('story')}
        >
          Story
        </button>
        <button 
          className={`tab-button ${activeTab === 'images' ? 'active' : ''}`}
          onClick={() => setActiveTab('images')}
        >
          Images
        </button>
        <button 
          className={`tab-button ${activeTab === 'audio' ? 'active' : ''}`}
          onClick={() => setActiveTab('audio')}
        >
          Audio
        </button>
      </div>
      
      <div className="detail-content">
        {activeTab === 'summary' && renderSummary()}
        {activeTab === 'story' && renderStory()}
        {activeTab === 'images' && renderImages()}
        {activeTab === 'audio' && renderAudio()}
      </div>
    </div>
  );
}

export default ProjectDetail; 