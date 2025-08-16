// Frontend Configuration
// This file manages environment variables and API endpoints

const config = {
    // API Configuration
    API_BASE_URL: process.env.API_BASE_URL || 'http://localhost:5000',
    API_ENDPOINTS: {
        SPOTIFY: '/api/spotify',
        CAMERA: '/api/camera'
    },
    
    // WebSocket Configuration
    WEBSOCKET_URL: process.env.WEBSOCKET_URL || 'http://localhost:5000',
    
    // App Configuration
    APP_NAME: 'Jupbox',
    APP_VERSION: '1.0.0',
    
    // UI Configuration
    UPDATE_INTERVAL: 1000, // 1 second
    NOTIFICATION_DURATION: 3000, // 3 seconds
    
    // Development Configuration
    DEV_MODE: process.env.NODE_ENV === 'development'
};

// Helper function to get full API URL
function getApiUrl(endpoint) {
    return `${config.API_BASE_URL}${endpoint}`;
}

// Helper function to get WebSocket URL
function getWebSocketUrl() {
    return config.WEBSOCKET_URL;
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { config, getApiUrl, getWebSocketUrl };
} else {
    // For browser environment
    window.config = config;
    window.getApiUrl = getApiUrl;
    window.getWebSocketUrl = getWebSocketUrl;
}
