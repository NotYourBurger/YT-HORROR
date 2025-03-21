import React, { useState, useEffect } from 'react';
import './Settings.css';

function Settings() {
  const [settings, setSettings] = useState({
    defaultVoice: 'en_emily',
    defaultVideoQuality: '4000k',
    defaultCinematicRatio: 2.35,
    useDustOverlay: true,
    saveOriginalFiles: true,
    darkMode: true
  });
  
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Fetch settings from API
    fetch('/api/settings')
      .then(res => res.json())
      .then(data => {
        if (data.settings) {
          setSettings(data.settings);
        }
      })
      .catch(err => {
        console.error('Error fetching settings:', err);
        setError('Failed to load settings. Using defaults.');
      })
      .finally(() => {
        setLoading(false);
      });
  }, []);

  const handleChange = (key, value) => {
    setSettings({
      ...settings,
      [key]: value
    });
    
    // Clear any previous saved state
    setSaved(false);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    setSaving(true);
    setError(null);
    setSaved(false);
    
    try {
      const response = await fetch('/api/settings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          settings: settings
        })
      });
      
      const data = await response.json();
      
      if (response.ok) {
        setSaved(true);
      } else {
        throw new Error(data.error || 'Failed to save settings');
      }
    } catch (err) {
      setError(err.message);
      console.error('Error saving settings:', err);
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="settings-page">
        <h1>Settings</h1>
        <div className="loading">Loading settings...</div>
      </div>
    );
  }

  return (
    <div className="settings-page">
      <h1>Application Settings</h1>
      
      {error && (
        <div className="error-message">
          <p>{error}</p>
          <button onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}
      
      {saved && (
        <div className="success-message">
          <p>Settings saved successfully!</p>
          <button onClick={() => setSaved(false)}>Dismiss</button>
        </div>
      )}
      
      <form onSubmit={handleSubmit}>
        <div className="settings-section">
          <h2>Default Generation Settings</h2>
          
          <div className="form-group">
            <label htmlFor="default-voice">Default Voice:</label>
            <select
              id="default-voice"
              value={settings.defaultVoice}
              onChange={(e) => handleChange('defaultVoice', e.target.value)}
            >
              <option value="en_emily">Emily (Female)</option>
              <option value="en_ryan">Ryan (Male)</option>
              <option value="en_ghost">Ghost (Ethereal)</option>
              <option value="en_whisper">Whisper (Creepy)</option>
            </select>
          </div>
          
          <div className="form-group">
            <label htmlFor="default-quality">Default Video Quality:</label>
            <select
              id="default-quality"
              value={settings.defaultVideoQuality}
              onChange={(e) => handleChange('defaultVideoQuality', e.target.value)}
            >
              <option value="2000k">Low (2000k)</option>
              <option value="4000k">Medium (4000k)</option>
              <option value="8000k">High (8000k)</option>
            </select>
          </div>
          
          <div className="form-group">
            <label htmlFor="default-ratio">Default Cinematic Ratio:</label>
            <select
              id="default-ratio"
              value={settings.defaultCinematicRatio}
              onChange={(e) => handleChange('defaultCinematicRatio', parseFloat(e.target.value))}
            >
              <option value="2.35">2.35:1 (Cinematic)</option>
              <option value="1.85">1.85:1 (Widescreen)</option>
              <option value="1.78">16:9 (Standard)</option>
            </select>
          </div>
          
          <div className="form-group checkbox-group">
            <input
              type="checkbox"
              id="use-dust"
              checked={settings.useDustOverlay}
              onChange={(e) => handleChange('useDustOverlay', e.target.checked)}
            />
            <label htmlFor="use-dust">Use dust overlay for cinematic effect</label>
          </div>
        </div>
        
        <div className="settings-section">
          <h2>Application Preferences</h2>
          
          <div className="form-group checkbox-group">
            <input
              type="checkbox"
              id="save-originals"
              checked={settings.saveOriginalFiles}
              onChange={(e) => handleChange('saveOriginalFiles', e.target.checked)}
            />
            <label htmlFor="save-originals">Save original files (audio, images) after video generation</label>
          </div>
          
          <div className="form-group checkbox-group">
            <input
              type="checkbox"
              id="dark-mode"
              checked={settings.darkMode}
              onChange={(e) => handleChange('darkMode', e.target.checked)}
            />
            <label htmlFor="dark-mode">Use dark mode</label>
          </div>
        </div>
        
        <div className="form-actions">
          <button 
            type="submit" 
            className="save-button"
            disabled={saving}
          >
            {saving ? 'Saving...' : 'Save Settings'}
          </button>
        </div>
      </form>
    </div>
  );
}

export default Settings; 