const { ipcRenderer } = require('electron');
const axios = require('axios');

// Backend API URL
const API_BASE = 'http://localhost:8000';

// Global variables
let selectedFiles = [];
let isRecording = false;
let recognition = null;

// Initialize the app
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    initializeSpeechRecognition();
});

// Event listeners
function initializeEventListeners() {
    // Tab switching
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', (e) => {
            const tabName = e.target.textContent.toLowerCase().includes('upload') ? 'upload' : 'chat';
            showTab(tabName);
        });
    });

    // File selection
    document.getElementById('selectFilesBtn').addEventListener('click', selectFiles);
    document.getElementById('uploadBtn').addEventListener('click', uploadFiles);

    // Chat functionality
    document.getElementById('sendBtn').addEventListener('click', sendMessage);
    document.getElementById('voiceBtn').addEventListener('click', toggleVoiceInput);
    
    // Enter key in textarea
    document.getElementById('queryInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Drag and drop
    const uploadArea = document.getElementById('uploadArea');
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    uploadArea.addEventListener('click', selectFiles);
}

// Tab switching
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active');
    });

    // Show selected tab
    document.getElementById(tabName).classList.add('active');
    document.querySelector(`[onclick="showTab('${tabName}')"]`).classList.add('active');
}

// File selection
async function selectFiles() {
    try {
        const filePaths = await ipcRenderer.invoke('select-files');
        if (filePaths && filePaths.length > 0) {
            selectedFiles = filePaths.map(path => ({
                path: path,
                name: path.split('\\').pop(),
                size: 0 // We'll get this when uploading
            }));
            updateFileList();
        }
    } catch (error) {
        showStatus('Error selecting files: ' + error.message, 'error');
    }
}

// Update file list display
function updateFileList() {
    const fileListElement = document.getElementById('fileList');
    const uploadBtn = document.getElementById('uploadBtn');

    if (selectedFiles.length === 0) {
        fileListElement.innerHTML = '';
        uploadBtn.disabled = true;
        return;
    }

    fileListElement.innerHTML = `
        <h3>Selected Files (${selectedFiles.length}):</h3>
        ${selectedFiles.map(file => `
            <div class="file-item">
                <span class="file-name">${file.name}</span>
                <span class="file-size">${formatFileSize(file.size)}</span>
            </div>
        `).join('')}
    `;

    uploadBtn.disabled = false;
}

// Upload files to backend
async function uploadFiles() {
    if (selectedFiles.length === 0) return;

    const uploadBtn = document.getElementById('uploadBtn');
    uploadBtn.disabled = true;
    showStatus('Uploading files...', 'loading');

    try {
        const formData = new FormData();
        
        // Read and append files
        for (const file of selectedFiles) {
            const response = await fetch(`file://${file.path}`);
            const blob = await response.blob();
            formData.append('files', blob, file.name);
        }

        const response = await axios.post(`${API_BASE}/upload/`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        });

        if (response.data.status === 'success') {
            showStatus(
                `Successfully uploaded ${response.data.files_processed} files. ` +
                `Total chunks processed: ${response.data.total_chunks}`,
                'success'
            );
            selectedFiles = [];
            updateFileList();
        }
    } catch (error) {
        showStatus('Upload failed: ' + error.message, 'error');
    } finally {
        uploadBtn.disabled = false;
    }
}

// Drag and drop handlers
function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
    e.currentTarget.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    
    const files = Array.from(e.dataTransfer.files).filter(file => 
        file.type === 'application/pdf' || file.name.endsWith('.pdf')
    );
    
    if (files.length > 0) {
        selectedFiles = files.map(file => ({
            path: file.path,
            name: file.name,
            size: file.size
        }));
        updateFileList();
    }
}

// Chat functionality
async function sendMessage() {
    const queryInput = document.getElementById('queryInput');
    const modelSelect = document.getElementById('modelSelect');
    const query = queryInput.value.trim();

    if (!query) return;

    // Add user message to chat
    addMessageToChat('user', query);
    queryInput.value = '';

    try {
        // Send request to backend
        const formData = new FormData();
        formData.append('query', query);
        formData.append('model', modelSelect.value);

        const response = await axios.post(`${API_BASE}/chat/`, formData);
        
        // Add assistant response to chat
        addMessageToChat('assistant', response.data.answer, response.data.model_used);
    } catch (error) {
        addMessageToChat('assistant', 'Sorry, I encountered an error: ' + error.message);
    }
}

// Add message to chat history
function addMessageToChat(sender, message, model = null) {
    const chatHistory = document.getElementById('chatHistory');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;

    const timestamp = new Date().toLocaleTimeString();
    const modelInfo = model ? ` (${model})` : '';

    messageDiv.innerHTML = `
        <div class="message-header">${sender === 'user' ? 'You' : 'AI Assistant'}${modelInfo} - ${timestamp}</div>
        <div class="message-content">${message}</div>
    `;

    chatHistory.appendChild(messageDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

// Speech recognition
function initializeSpeechRecognition() {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        recognition.onresult = function(event) {
            const transcript = event.results[0][0].transcript;
            document.getElementById('queryInput').value = transcript;
            stopRecording();
        };

        recognition.onerror = function(event) {
            console.error('Speech recognition error:', event.error);
            stopRecording();
        };

        recognition.onend = function() {
            stopRecording();
        };
    } else {
        document.getElementById('voiceBtn').style.display = 'none';
        console.log('Speech recognition not supported');
    }
}

function toggleVoiceInput() {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}

function startRecording() {
    if (recognition) {
        isRecording = true;
        recognition.start();
        document.getElementById('voiceBtn').classList.add('recording');
        document.getElementById('queryInput').placeholder = 'Listening...';
    }
}

function stopRecording() {
    if (recognition) {
        isRecording = false;
        recognition.stop();
        document.getElementById('voiceBtn').classList.remove('recording');
        document.getElementById('queryInput').placeholder = 'Type your question here...';
    }
}

// Utility functions
function showStatus(message, type) {
    const statusElement = document.getElementById('uploadStatus');
    statusElement.innerHTML = message;
    statusElement.className = `status ${type}`;
    
    // Auto-hide success messages after 5 seconds
    if (type === 'success') {
        setTimeout(() => {
            statusElement.innerHTML = '';
            statusElement.className = 'status';
        }, 5000);
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return 'Unknown size';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Make showTab function global for onclick handlers
window.showTab = showTab;
