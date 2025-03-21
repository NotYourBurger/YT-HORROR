import React, { useState, useEffect } from 'react';
import './ModelManager.css';

function ModelManager() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [apiKeys, setApiKeys] = useState({});

  useEffect(() => {
    fetchModels();
    
    // Set up polling for model download status
    const intervalId = setInterval(() => {
      // Only poll if there are models being downloaded
      if (models.some(model => 
        model.status === 'downloading' || 
        model.status === 'extracting'
      )) {
        fetchModels();
      }
    }, 2000);
    
    return () => clearInterval(intervalId);
  }, [models]);

  const fetchModels = async () => {
    try {
      const response = await fetch('/api/models');
      const data = await response.json();
      
      // Convert to array and add status info
      const modelArray = Object.entries(data.models).map(([id, details]) => ({
        id,
        ...details,
        status: details.installed ? 'installed' : 'not_installed',
        progress: 0
      }));
      
      // Update with current download status
      for (const model of modelArray) {
        if (!model.installed) {
          await fetchModelStatus(model);
        }
      }
      
      setModels(modelArray);
    } catch (err) {
      setError('Failed to load models. Please try again later.');
      console.error('Error fetching models:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchModelStatus = async (model) => {
    if (!model.download_url && !model.api_key_required) return;
    
    try {
      const response = await fetch(`/api/models/${model.id}/status`);
      const status = await response.json();
      
      if (status.status) {
        model.status = status.status;
        model.progress = status.progress || 0;
        model.message = status.message || '';
      }
    } catch (err) {
      console.error(`Error fetching status for ${model.id}:`, err);
    }
  };

  const handleDownload = async (modelId) => {
    try {
      const model = models.find(m => m.id === modelId);
      if (!model) return;
      
      model.status = 'starting';
      setModels([...models]);
      
      const response = await fetch(`/api/models/${modelId}/download`, {
        method: 'POST'
      });
      
      const data = await response.json();
      if (data.status === 'download_started') {
        // Update the model in our state
        const updatedModels = models.map(m => 
          m.id === modelId 
            ? { ...m, status: 'downloading', progress: 0 } 
            : m
        );
        setModels(updatedModels);
      } else {
        throw new Error(data.message || 'Download failed to start');
      }
    } catch (err) {
      setError(`Failed to start download: ${err.message}`);
      console.error('Error starting download:', err);
      
      // Reset status
      const updatedModels = models.map(m => 
        m.id === modelId 
          ? { ...m, status: 'not_installed', progress: 0 } 
          : m
      );
      setModels(updatedModels);
    }
  };

  const handleSaveApiKey = async (modelId) => {
    try {
      const apiKey = apiKeys[modelId];
      if (!apiKey) {
        setError('Please enter an API key');
        return;
      }
      
      const response = await fetch(`/api/models/${modelId}/api-key`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ api_key: apiKey })
      });
      
      const data = await response.json();
      if (data.success) {
        // Update the model in our state
        const updatedModels = models.map(m => 
          m.id === modelId 
            ? { ...m, installed: true, status: 'installed' } 
            : m
        );
        setModels(updatedModels);
        
        // Clear the API key from state
        setApiKeys({
          ...apiKeys,
          [modelId]: ''
        });
      } else {
        throw new Error(data.message || 'Failed to save API key');
      }
    } catch (err) {
      setError(`Failed to save API key: ${err.message}`);
      console.error('Error saving API key:', err);
    }
  };

  const handleApiKeyChange = (modelId, value) => {
    setApiKeys({
      ...apiKeys,
      [modelId]: value
    });
  };

  if (loading) {
    return (
      <div className="model-manager">
        <h1>AI Model Manager</h1>
        <div className="loading">Loading models...</div>
      </div>
    );
  }

  return (
    <div className="model-manager">
      <h1>AI Model Manager</h1>
      
      {error && (
        <div className="error-message">
          <p>{error}</p>
          <button onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}
      
      <p className="description">
        This application requires various AI models to function. 
        You can download the models here or provide API keys for cloud-based services.
      </p>
      
      <div className="models-grid">
        {models.map(model => (
          <div className="model-card" key={model.id}>
            <div className="model-header">
              <h2>{model.name}</h2>
              <span className={`status-badge ${model.status}`}>
                {model.status === 'installed' ? 'Installed' : 
                 model.status === 'downloading' ? 'Downloading...' :
                 model.status === 'extracting' ? 'Extracting...' :
                 model.status === 'error' ? 'Error' : 'Not Installed'}
              </span>
            </div>
            
            <p className="model-description">{model.description}</p>
            
            <div className="model-details">
              <div>Version: {model.version}</div>
              {model.size_mb > 0 && <div>Size: {model.size_mb} MB</div>}
            </div>
            
            {model.status === 'downloading' && (
              <div className="progress-bar-container">
                <div 
                  className="progress-bar" 
                  style={{ width: `${model.progress}%` }}
                ></div>
                <span>{model.progress}%</span>
              </div>
            )}
            
            {model.message && <p className="status-message">{model.message}</p>}
            
            {!model.installed && model.api_key_required && (
              <div className="api-key-container">
                <input
                  type="password"
                  placeholder="Enter API Key"
                  value={apiKeys[model.id] || ''}
                  onChange={(e) => handleApiKeyChange(model.id, e.target.value)}
                />
                <button 
                  onClick={() => handleSaveApiKey(model.id)}
                  disabled={!apiKeys[model.id]}
                >
                  Save API Key
                </button>
              </div>
            )}
            
            {!model.installed && model.download_url && (
              <button 
                className="download-button"
                onClick={() => handleDownload(model.id)}
                disabled={model.status === 'downloading' || model.status === 'extracting'}
              >
                {model.status === 'downloading' ? 'Downloading...' : 
                 model.status === 'extracting' ? 'Extracting...' : 'Download Model'}
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export default ModelManager; 