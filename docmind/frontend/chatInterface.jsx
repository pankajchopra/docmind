// src/components/ChatInterface.jsx
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [retrievalType, setRetrievalType] = useState('hybrid');
  const [files, setFiles] = useState([]);
  const messagesEndRef = useRef(null);

  // Scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;
    
    const userMessage = input;
    setInput('');
    
    // Add user message to chat
    setMessages((prev) => [...prev, { text: userMessage, sender: 'user' }]);
    
    // Show loading indicator
    setIsLoading(true);
    
    try {
      // Send message to API
      const response = await axios.post(`${API_URL}/chat`, {
        message: userMessage,
        retrieval_type: retrievalType
      });
      
      // Add response to chat
      setMessages((prev) => [...prev, { text: response.data.response, sender: 'bot' }]);
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages((prev) => [...prev, { 
        text: 'Sorry, there was an error processing your request.', 
        sender: 'bot' 
      }]);
    }
    
    setIsLoading(false);
  };

  const handleFileUpload = async (e) => {
    const selectedFiles = Array.from(e.target.files);
    setFiles(selectedFiles);
  };

  const uploadFiles = async () => {
    if (files.length === 0) return;
    
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });
    
    try {
      const response = await axios.post(`${API_URL}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      alert(response.data.message);
      setFiles([]);
    } catch (error) {
      console.error('Error uploading files:', error);
      alert('Error uploading files');
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h1>DocMind: Intelligent Document Search</h1>
      </div>
      
      <div className="main-content">
        <div className="chat-area">
          <div className="messages">
            {messages.map((msg, index) => (
              <div key={index} className={`message ${msg.sender}`}>
                {msg.text}
              </div>
            ))}
            {isLoading && (
              <div className="message bot loading">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          
          <div className="input-area">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask me anything about your documents..."
              onKeyPress={(e) => e.key === 'Enter' && handleSend()}
            />
            <button onClick={handleSend} disabled={isLoading}>Send</button>
          </div>
        </div>
        
        <div className="sidebar">
          <div className="search-options">
            <h3>Search Options</h3>
            <div className="retrieval-options">
              <label>Retrieval Method:</label>
              <div className="radio-group">
                {['hybrid', 'graph', 'transformed'].map(type => (
                  <label key={type}>
                    <input
                      type="radio"
                      name="retrieval"
                      value={type}
                      checked={retrievalType === type}
                      onChange={() => setRetrievalType(type)}
                    />
                    {type}
                  </label>
                ))}
              </div>
            </div>
            
            <div className="file-upload">
              <h3>Upload Documents</h3>
              <input 
                type="file" 
                multiple 
                accept=".pdf,.txt,.docx" 
                onChange={handleFileUpload} 
              />
              <div>
                {files.length > 0 && (
                  <>
                    <p>{files.length} files selected</p>
                    <button onClick={uploadFiles}>Upload</button>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;