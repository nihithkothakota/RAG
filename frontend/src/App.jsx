import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import './index.css';

function App() {
  const [messages, setMessages] = useState([
    { type: 'bot', text: 'Hello! Please upload a document to get started.' }
  ]);
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      await uploadFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = async (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      await uploadFile(e.target.files[0]);
    }
  };

  const uploadFile = async (file) => {
    setUploadStatus('Uploading...');
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      if (res.ok) {
        setUploadStatus(`Success! ${file.name} ready.`);
        setMessages(prev => [...prev, { type: 'bot', text: `I've analyzed ${file.name}. What would you like to know about it?` }]);
      } else {
        setUploadStatus(`Error: ${data.detail}`);
      }
    } catch (err) {
      setUploadStatus("Error connecting to server.");
    }
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    const userQuery = query;
    // Capture current history before adding the new user message
    const currentHistory = [...messages];
    
    setMessages(prev => [...prev, { type: 'user', text: userQuery }]);
    setQuery('');
    setIsLoading(true);

    try {
      const res = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          query: userQuery,
          history: currentHistory
        })
      });
      const data = await res.json();
      if (res.ok) {
        setMessages(prev => [...prev, { type: 'bot', text: data.answer }]);
      } else {
        setMessages(prev => [...prev, { type: 'bot', text: `Error: ${data.detail}` }]);
      }
    } catch (err) {
      setMessages(prev => [...prev, { type: 'bot', text: "Error connecting to server." }]);
    }
    setIsLoading(false);
  };

  return (
    <div className="app-container">
      <div className="sidebar glass">
        <div className="header">
          <h1>HPC-AI RAG</h1>
          <p>Local document intelligence</p>
        </div>
        
        <div 
          className={`upload-zone ${dragActive ? 'drag-active' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input type="file" onChange={handleChange} accept=".txt,.pdf" />
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
          <p style={{ color: 'var(--text-muted)' }}>Drag & Drop or Click<br/>to upload PDF/TXT</p>
        </div>
        
        {uploadStatus && (
          <div className={`status-indicator ${uploadStatus.startsWith('Error') ? 'error' : ''}`}>
            {uploadStatus}
          </div>
        )}
      </div>

      <div className="chat-container glass">
        <div className="chat-history">
          {messages.map((msg, idx) => (
            <div key={idx} className={`chat-message ${msg.type}`}>
              {msg.type === 'bot' ? (
                <ReactMarkdown>{msg.text}</ReactMarkdown>
              ) : (
                msg.text
              )}
            </div>
          ))}
          {isLoading && (
            <div className="chat-message bot">
              <span className="typing-dot"></span>
              <span className="typing-dot" style={{ margin: '0 4px' }}></span>
              <span className="typing-dot"></span>
            </div>
          )}
        </div>
        
        <form className="chat-input-container" onSubmit={sendMessage}>
          <input 
            type="text" 
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question about your document..." 
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading || !query.trim()}>
            Send
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;
