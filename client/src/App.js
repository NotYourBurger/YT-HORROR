import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import ModelManager from './pages/ModelManager';
import ProjectList from './pages/ProjectList';
import ProjectDetail from './pages/ProjectDetail';
import NewProject from './pages/NewProject';
import Settings from './pages/Settings';
import './App.css';

function App() {
  const [isLoading, setIsLoading] = useState(true);
  const [apiStatus, setApiStatus] = useState(null);

  useEffect(() => {
    // Check API connection on startup
    fetch('/api/status')
      .then(response => response.json())
      .then(data => {
        setApiStatus(data.status === 'ok' ? 'connected' : 'error');
      })
      .catch(error => {
        console.error('Error connecting to API:', error);
        setApiStatus('error');
      })
      .finally(() => {
        setIsLoading(false);
      });
  }, []);

  if (isLoading) {
    return (
      <div className="app-loading">
        <div className="loading-spinner"></div>
        <p>Initializing application...</p>
      </div>
    );
  }

  if (apiStatus === 'error') {
    return (
      <div className="app-error">
        <h1>Connection Error</h1>
        <p>Unable to connect to the API server. Please make sure the server is running.</p>
        <button onClick={() => window.location.reload()}>Retry</button>
      </div>
    );
  }

  return (
    <Router>
      <div className="app">
        <Navbar />
        <div className="content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/models" element={<ModelManager />} />
            <Route path="/projects" element={<ProjectList />} />
            <Route path="/projects/:id" element={<ProjectDetail />} />
            <Route path="/new-project" element={<NewProject />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App; 