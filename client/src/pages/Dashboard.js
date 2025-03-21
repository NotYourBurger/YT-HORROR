import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import './Dashboard.css';

function Dashboard() {
  const [modelStatus, setModelStatus] = useState({});
  const [recentProjects, setRecentProjects] = useState([]);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    totalProjects: 0,
    completedVideos: 0,
    storiesFetched: 0
  });

  useEffect(() => {
    Promise.all([
      fetch('/api/models').then(res => res.json()),
      fetch('/api/projects?limit=3').then(res => res.json()),
      fetch('/api/stats').then(res => res.json())
    ])
      .then(([modelsData, projectsData, statsData]) => {
        // Process models data
        const status = {
          gemini: modelsData.models.gemini.installed,
          whisper: modelsData.models.whisper.installed,
          tts: modelsData.models.tts.installed,
          stable_diffusion: modelsData.models.stable_diffusion.installed,
          sound_effects: modelsData.models.sound_effects.installed
        };
        setModelStatus(status);
        
        // Set recent projects
        setRecentProjects(projectsData.projects || []);
        
        // Set stats
        setStats(statsData);
      })
      .catch(err => {
        console.error('Error fetching dashboard data:', err);
      })
      .finally(() => {
        setLoading(false);
      });
  }, []);

  const getMissingModels = () => {
    return Object.entries(modelStatus)
      .filter(([_, installed]) => !installed)
      .map(([modelId]) => modelId);
  };

  if (loading) {
    return (
      <div className="dashboard">
        <h1>Dashboard</h1>
        <div className="loading">Loading dashboard data...</div>
      </div>
    );
  }

  return (
    <div className="dashboard">
      <h1>Horror Story Generator Dashboard</h1>
      
      <div className="dashboard-grid">
        {getMissingModels().length > 0 && (
          <div className="dashboard-card model-warning">
            <h2>Model Installation Required</h2>
            <p>You need to install the following models before generating videos:</p>
            <ul>
              {getMissingModels().map(modelId => (
                <li key={modelId}>{modelId}</li>
              ))}
            </ul>
            <Link to="/models" className="card-button">Go to Models</Link>
          </div>
        )}
        
        <div className="dashboard-card stats-card">
          <h2>Generation Statistics</h2>
          <div className="stats-grid">
            <div className="stat-item">
              <span className="stat-number">{stats.totalProjects}</span>
              <span className="stat-label">Total Projects</span>
            </div>
            <div className="stat-item">
              <span className="stat-number">{stats.completedVideos}</span>
              <span className="stat-label">Videos Created</span>
            </div>
            <div className="stat-item">
              <span className="stat-number">{stats.storiesFetched}</span>
              <span className="stat-label">Stories Fetched</span>
            </div>
          </div>
        </div>
        
        <div className="dashboard-card new-project-card">
          <h2>Create New Horror Video</h2>
          <p>Generate a new cinematic horror story video with AI narration and visuals.</p>
          <Link to="/new-project" className="card-button">Start New Project</Link>
        </div>
        
        {recentProjects.length > 0 && (
          <div className="dashboard-card recent-projects">
            <h2>Recent Projects</h2>
            <div className="recent-projects-list">
              {recentProjects.map(project => (
                <div className="recent-project-item" key={project.id}>
                  <h3>{project.title}</h3>
                  <p>Created: {new Date(project.date_created).toLocaleDateString()}</p>
                  <span className={`project-status ${project.status}`}>{project.status}</span>
                  <Link to={`/projects/${project.id}`} className="view-project">View Project</Link>
                </div>
              ))}
            </div>
            <Link to="/projects" className="card-button secondary">View All Projects</Link>
          </div>
        )}
      </div>
    </div>
  );
}

export default Dashboard; 