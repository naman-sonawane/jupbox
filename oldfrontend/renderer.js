
const SPOTIFY_API_BASE = 'http://localhost:5000/api/spotify';


let socket = null;


let currentTrack = null;
let updateInterval = null;
let userInfo = null;


const elements = {
    
    userName: document.getElementById('userName'),
    userImage: document.getElementById('userImage'),
    
    
    trackName: document.getElementById('trackName'),
    trackArtist: document.getElementById('trackArtist'),
    trackAlbum: document.getElementById('trackAlbum'),
    albumCover: document.getElementById('albumCover'),
    noCover: document.getElementById('noCover'),
    
    
    progressFill: document.getElementById('progressFill'),
    currentTime: document.getElementById('currentTime'),
    totalTime: document.getElementById('totalTime'),
    
    
    playPauseBtn: document.getElementById('playPauseBtn'),
    playIcon: document.getElementById('playIcon'),
    prevBtn: document.getElementById('prevBtn'),
    nextBtn: document.getElementById('nextBtn'),
    
    
    statusIndicator: document.getElementById('statusIndicator'),
    statusText: document.getElementById('statusText'),
    
    
    searchInput: document.getElementById('searchInput'),
    searchBtn: document.getElementById('searchBtn'),
    searchResults: document.getElementById('searchResults'),
    
    
    platform: document.getElementById('platform'),
    nodeVersion: document.getElementById('nodeVersion'),
    electronVersion: document.getElementById('electronVersion')
};


document.addEventListener('DOMContentLoaded', () => {
    console.log('🎵 Spotify Electron app loaded!');
    
    initializeSpotify();
    initializeWebSocket();
    setupEventListeners();
    displaySystemInfo();
    addInteractiveFeatures();
});


function initializeWebSocket() {
    try {
        socket = io('http://localhost:5000');
        
        socket.on('connect', () => {
            console.log('🔌 Connected to camera control WebSocket');
            showNotification('📹 Camera Control Connected', 'Gesture detection is now active!');
            
            
            updateCameraStatus('Connected', 'connected');
            
            
            socket.emit('request_playback_state');
        });
        
        socket.on('disconnect', () => {
            console.log('🔌 Disconnected from camera control WebSocket');
            showNotification('📹 Camera Control Disconnected', 'Gesture detection is offline');
            
            
            updateCameraStatus('Disconnected', 'error');
        });
        
        socket.on('playback_state_changed', (data) => {
            console.log('🎵 Playback state changed via camera:', data);
            
            
            updatePlayPauseButton(data.is_playing);
            
            
            const action = data.is_playing ? '▶️ Playing' : '⏸️ Paused';
            showNotification('📹 Camera Control', `Music ${action} via gesture detection`);
            
            
            setTimeout(updateCurrentTrack, 500);
        });
        
        socket.on('playback_state', (data) => {
            console.log('🎵 Current playback state:', data);
            if (!data.error) {
                updatePlayPauseButton(data.is_playing);
            }
        });
        
        socket.on('connected', (data) => {
            console.log('📹 Camera control system message:', data.message);
        });
        
    } catch (error) {
        console.error('WebSocket connection error:', error);
        showNotification('❌ WebSocket Error', 'Could not connect to camera control system');
    }
}


async function initializeSpotify() {
    try {
        updateConnectionStatus('Connecting to Spotify...', 'connecting');
        
        
        const userResponse = await fetch(`${SPOTIFY_API_BASE}/user`);
        if (userResponse.ok) {
            const userData = await userResponse.json();
            if (userData.success) {
                userInfo = userData.user;
                displayUserInfo(userInfo);
                updateConnectionStatus('Connected to Spotify', 'connected');
                
                
                startPlaybackTracking();
                
                showNotification('🎵 Connected to Spotify!', `Welcome back, ${userInfo.name}!`);
            } else {
                throw new Error(userData.error || 'Failed to get user info');
            }
        } else {
            throw new Error('Failed to connect to Spotify API');
        }
    } catch (error) {
        console.error('Spotify connection error:', error);
        updateConnectionStatus('Connection failed', 'error');
        showNotification('❌ Spotify Connection Failed', 'Make sure the backend server is running and Spotify is authorized.');
    }
}


function setupEventListeners() {
    
    elements.playPauseBtn.addEventListener('click', togglePlayPause);
    elements.prevBtn.addEventListener('click', previousTrack);
    elements.nextBtn.addEventListener('click', nextTrack);
    
    
    elements.searchBtn.addEventListener('click', performSearch);
    elements.searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            performSearch();
        }
    });
    
    
    document.addEventListener('keydown', handleKeyboardShortcuts);
}


async function togglePlayPause() {
    try {
        const response = await fetch(`${SPOTIFY_API_BASE}/play`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        if (response.ok) {
            const data = await response.json();
            if (data.success) {
                
                updatePlayPauseButton(data.action === 'playing');
                
                setTimeout(updateCurrentTrack, 500);
            }
        }
    } catch (error) {
        console.error('Play/pause error:', error);
        showNotification('❌ Playback Error', 'Failed to control playback');
    }
}

async function previousTrack() {
    try {
        const response = await fetch(`${SPOTIFY_API_BASE}/previous`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        if (response.ok) {
            setTimeout(updateCurrentTrack, 500);
        }
    } catch (error) {
        console.error('Previous track error:', error);
    }
}

async function nextTrack() {
    try {
        const response = await fetch(`${SPOTIFY_API_BASE}/next`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        if (response.ok) {
            setTimeout(updateCurrentTrack, 500);
        }
    } catch (error) {
        console.error('Next track error:', error);
    }
}


async function updateCurrentTrack() {
    try {
        const response = await fetch(`${SPOTIFY_API_BASE}/current`);
        if (response.ok) {
            const data = await response.json();
            if (data.success) {
                if (data.track) {
                    currentTrack = data.track;
                    displayTrackInfo(currentTrack);
                    updatePlayPauseButton(currentTrack.is_playing);
                } else {
                    displayNoTrack();
                }
            }
        }
    } catch (error) {
        console.error('Update track error:', error);
    }
}

function displayTrackInfo(track) {
    elements.trackName.textContent = track.name;
    elements.trackArtist.textContent = track.artist;
    elements.trackAlbum.textContent = track.album;
    
    if (track.cover_url) {
        elements.albumCover.src = track.cover_url;
        elements.albumCover.style.display = 'block';
        elements.noCover.style.display = 'none';
    } else {
        elements.albumCover.style.display = 'none';
        elements.noCover.style.display = 'flex';
    }
    
    
    updateProgress(track.progress_ms, track.duration_ms);
}

function displayNoTrack() {
    elements.trackName.textContent = 'No track playing';
    elements.trackArtist.textContent = 'Start playing music on Spotify';
    elements.trackAlbum.textContent = '';
    elements.albumCover.style.display = 'none';
    elements.noCover.style.display = 'flex';
    elements.progressFill.style.width = '0%';
    elements.currentTime.textContent = '0:00';
    elements.totalTime.textContent = '0:00';
}

function updateProgress(current, total) {
    if (total > 0) {
        const percentage = (current / total) * 100;
        elements.progressFill.style.width = `${percentage}%`;
        elements.currentTime.textContent = formatTime(current);
        elements.totalTime.textContent = formatTime(total);
    }
}

function formatTime(ms) {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

function updatePlayPauseButton(isPlaying) {
    elements.playIcon.textContent = isPlaying ? '⏸' : '▶';
}

function displayUserInfo(user) {
    elements.userName.textContent = user.name;
    if (user.image) {
        elements.userImage.src = user.image;
        elements.userImage.style.display = 'block';
    }
}

function updateConnectionStatus(message, status) {
    elements.statusText.textContent = message;
    elements.statusIndicator.className = `status-indicator ${status}`;
}

function updateCameraStatus(message, status) {
    const cameraStatusElement = document.getElementById('cameraStatus');
    if (cameraStatusElement) {
        cameraStatusElement.textContent = message;
        cameraStatusElement.className = status;
    }
}


async function performSearch() {
    const query = elements.searchInput.value.trim();
    if (!query) return;
    
    try {
        elements.searchResults.innerHTML = '<p>Searching...</p>';
        
        const response = await fetch(`${SPOTIFY_API_BASE}/search?q=${encodeURIComponent(query)}`);
        if (response.ok) {
            const data = await response.json();
            if (data.success) {
                displaySearchResults(data.tracks);
            } else {
                elements.searchResults.innerHTML = '<p>Search failed</p>';
            }
        }
    } catch (error) {
        console.error('Search error:', error);
        elements.searchResults.innerHTML = '<p>Search failed</p>';
    }
}

function displaySearchResults(tracks) {
    if (tracks.length === 0) {
        elements.searchResults.innerHTML = '<p>No tracks found</p>';
        return;
    }
    
    elements.searchResults.innerHTML = tracks.map(track => `
        <div class="search-result-item">
            <img src="${track.cover_url || ''}" alt="Cover" class="search-result-cover" onerror="this.style.display='none'">
            <div class="search-result-info">
                <h4>${track.name}</h4>
                <p>${track.artist} • ${track.album}</p>
            </div>
            <button class="play-result-btn" onclick="playTrack('${track.uri}')">Play</button>
        </div>
    `).join('');
}

async function playTrack(uri) {
    try {
        const response = await fetch(`${SPOTIFY_API_BASE}/play-track`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ uri })
        });
        
        if (response.ok) {
            setTimeout(updateCurrentTrack, 1000);
            showNotification('🎵 Playing Track', 'Track started successfully!');
        }
    } catch (error) {
        console.error('Play track error:', error);
        showNotification('❌ Playback Error', 'Failed to play track');
    }
}


function startPlaybackTracking() {
    
    updateCurrentTrack();
    
    
    updateInterval = setInterval(updateCurrentTrack, 2000);
}


function displaySystemInfo() {
    elements.platform.textContent = navigator.platform;
    elements.nodeVersion.textContent = 'Available via IPC';
    elements.electronVersion.textContent = 'Available via IPC';
    
    [elements.platform, elements.nodeVersion, elements.electronVersion].forEach(el => {
        el.classList.add('loading');
    });
    
    setTimeout(() => {
        [elements.platform, elements.nodeVersion, elements.electronVersion].forEach(el => {
            el.classList.remove('loading');
        });
    }, 2000);
}


function addInteractiveFeatures() {
    const cards = document.querySelectorAll('.card');
    
    cards.forEach(card => {
        card.addEventListener('mouseenter', () => {
            card.style.transform = 'translateY(-5px) scale(1.02)';
        });
        
        card.addEventListener('mouseleave', () => {
            card.style.transform = 'translateY(0) scale(1)';
        });
    });
}


function handleKeyboardShortcuts(e) {
    
    if (e.code === 'Space' && e.target.tagName !== 'INPUT') {
        e.preventDefault();
        togglePlayPause();
    }
    
    
    if (e.code === 'ArrowLeft' && e.ctrlKey) {
        e.preventDefault();
        previousTrack();
    }
    
    if (e.code === 'ArrowRight' && e.ctrlKey) {
        e.preventDefault();
        nextTrack();
    }
    
    
    if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
        e.preventDefault();
        location.reload();
    }
    
    
    if (e.key === 'Escape') {
        const notifications = document.querySelectorAll('[style*="position: fixed"]');
        notifications.forEach(notification => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        });
    }
}


function showNotification(title, message) {
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: white;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
        z-index: 1000;
        transform: translateX(100%);
        transition: transform 0.3s ease;
        max-width: 300px;
    `;
    
    notification.innerHTML = `
        <h4 style="margin: 0 0 5px 0; color: #667eea;">${title}</h4>
        <p style="margin: 0; color: #666; font-size: 14px;">${message}</p>
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
    }, 100);
    
    setTimeout(() => {
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 5000);
}


window.addEventListener('beforeunload', () => {
    if (updateInterval) {
        clearInterval(updateInterval);
    }
});
