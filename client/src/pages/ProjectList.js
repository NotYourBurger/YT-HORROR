import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import './ProjectList.css';

function ProjectList() {
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filter, setFilter] = useState('all');

  useEffect(() => {
    fetchProjects();
  }, []);

  const fetchProjects = async () => {
    try {
      const response = await fetch('/api/projects');
      const data = await response.json();
      
      if (response.ok) {
        setProjects(data.projects || []);
      } else {
        throw new Error(data.error || 'Failed to load projects');
      }
    } catch (err) {
      setError(err.message);
      console.error('Error fetching projects:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteProject = async (projectId) => {
    if (!window.confirm('Are you sure you want to delete this project? This action cannot be undone.')) {
      return;
    }
    
    try {
      const response = await fetch(`/api/projects/${projectId}`, {
        method: 'DELETE'
      });
      
      if (response.ok) {
        // Remove the project from state
        setProjects(projects.filter(p => p.id !== projectId));
      } else {
        const data = await response.json();
        throw new Error(data.error || 'Failed to delete project');
      }
    } catch (err) {
      setError(err.message);
      console.error('Error deleting project:', err);
    }
  };

  const filteredProjects = () => {
    if (filter === 'all') {
      return projects;
    }
    return projects.filter(project => project.status === filter);
  };

  const renderProjects = () => {
    const filtered = filteredProjects();
    
    if (filtered.length === 0) {
      return (
        <div className="no-projects">
          <p>No projects found. Create a new project to get started!</p>
          <Link to="/new-project" className="primary-button">Create New Project</Link>
        </div>
      );
    }
    
    return (
      <div className="project-grid">
        {filtered.map(project => (
          <div className="project-card" key={project.id}>
            <div className="project-header">
              <h2>{project.title}</h2>
              <span className={`project-status ${project.status}`}>
                {project.status === 'completed' ? 'Completed' :
                 project.status === 'processing' ? 'Processing' :
                 project.status === 'failed' ? 'Failed' : 'Draft'}
              </span>
            </div>
            
            <div className="project-details">
              <div>Created: {new Date(project.date_created).toLocaleDateString()}</div>
              <div>Source: r/{project.subreddit}</div>
            </div>
            
            <div className="project-actions">
              <Link to={`/projects/${project.id}`} className="action-button view">
                View Details
              </Link>
              <button 
                className="action-button delete"
                onClick={() => handleDeleteProject(project.id)}
              >
                Delete
              </button>
            </div>
          </div>
        ))}
      </div>
    );
  };

  if (loading) {
    return (
      <div className="project-list">
        <h1>Your Projects</h1>
        <div className="loading">Loading projects...</div>
      </div>
    );
  }

  return (
    <div className="project-list">
      <div className="page-header">
        <h1>Your Projects</h1>
        <Link to="/new-project" className="new-project-button">
          Create New Project
        </Link>
      </div>
      
      {error && (
        <div className="error-message">
          <p>{error}</p>
          <button onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}
      
      <div className="filter-controls">
        <span>Filter:</span>
        <button 
          className={`filter-button ${filter === 'all' ? 'active' : ''}`}
          onClick={() => setFilter('all')}
        >
          All
        </button>
        <button 
          className={`filter-button ${filter === 'completed' ? 'active' : ''}`}
          onClick={() => setFilter('completed')}
        >
          Completed
        </button>
        <button 
          className={`filter-button ${filter === 'processing' ? 'active' : ''}`}
          onClick={() => setFilter('processing')}
        >
          Processing
        </button>
        <button 
          className={`filter-button ${filter === 'failed' ? 'active' : ''}`}
          onClick={() => setFilter('failed')}
        >
          Failed
        </button>
      </div>
      
      {renderProjects()}
    </div>
  );
}

export default ProjectList; 