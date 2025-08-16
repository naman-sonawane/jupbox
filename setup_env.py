#!/usr/bin/env python3
"""
Environment Setup Script for Jupbox
This script helps you set up your .env file with the required configuration.
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """Create a .env file with user input"""
    
    print("🎵 Jupbox Environment Setup")
    print("=" * 40)
    print("This script will help you create your .env file with all required configuration.")
    print()
    
    # Check if .env already exists
    env_path = Path('.env')
    if env_path.exists():
        response = input("⚠️  .env file already exists. Overwrite? (y/N): ").lower()
        if response != 'y':
            print("❌ Setup cancelled.")
            return False
    
    # Collect configuration
    print("\n📝 Please provide the following configuration:")
    print()
    
    # Spotify Configuration
    print("🎵 Spotify API Configuration:")
    spotify_client_id = input("Spotify Client ID: ").strip()
    spotify_client_secret = input("Spotify Client Secret: ").strip()
    spotify_redirect_uri = input("Spotify Redirect URI (default: http://127.0.0.1:8888/callback): ").strip()
    if not spotify_redirect_uri:
        spotify_redirect_uri = "http://127.0.0.1:8888/callback"
    
    # Roboflow Configuration
    print("\n🤖 Roboflow API Configuration:")
    roboflow_api_key = input("Roboflow API Key: ").strip()
    roboflow_model_id = input("Roboflow Model ID (default: numbers-qysva/7): ").strip()
    if not roboflow_model_id:
        roboflow_model_id = "numbers-qysva/7"
    
    # Server Configuration
    print("\n🌐 Server Configuration:")
    flask_host = input("Flask Host (default: 0.0.0.0): ").strip()
    if not flask_host:
        flask_host = "0.0.0.0"
    
    flask_port = input("Flask Port (default: 5000): ").strip()
    if not flask_port:
        flask_port = "5000"
    
    flask_debug = input("Flask Debug Mode (True/False, default: True): ").strip()
    if not flask_debug:
        flask_debug = "True"
    
    # Webcam Configuration
    print("\n📹 Webcam Configuration:")
    webcam_index = input("Webcam Index (default: 0): ").strip()
    if not webcam_index:
        webcam_index = "0"
    
    # Create .env content
    env_content = f"""# Spotify API Configuration
SPOTIFY_CLIENT_ID={spotify_client_id}
SPOTIFY_CLIENT_SECRET={spotify_client_secret}
SPOTIFY_REDIRECT_URI={spotify_redirect_uri}

# Roboflow API Configuration
ROBOFLOW_API_KEY={roboflow_api_key}
ROBOFLOW_MODEL_ID={roboflow_model_id}

# Flask Server Configuration
FLASK_HOST={flask_host}
FLASK_PORT={flask_port}
FLASK_DEBUG={flask_debug}

# Webcam Configuration
WEBCAM_INDEX={webcam_index}
"""
    
    # Write .env file
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print(f"\n✅ .env file created successfully!")
        print(f"📁 Location: {env_path.absolute()}")
        return True
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

def validate_env_file():
    """Validate that the .env file has all required variables"""
    env_path = Path('.env')
    if not env_path.exists():
        print("❌ .env file not found!")
        return False
    
    required_vars = [
        'SPOTIFY_CLIENT_ID',
        'SPOTIFY_CLIENT_SECRET',
        'ROBOFLOW_API_KEY'
    ]
    
    missing_vars = []
    with open('.env', 'r') as f:
        content = f.read()
        for var in required_vars:
            if f"{var}=" not in content:
                missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    print("✅ .env file validation passed!")
    return True

def main():
    """Main setup function"""
    if len(sys.argv) > 1 and sys.argv[1] == '--validate':
        validate_env_file()
        return
    
    if create_env_file():
        print("\n🎉 Setup complete! You can now run the application.")
        print("\n📖 Next steps:")
        print("1. Make sure you have the required API keys")
        print("2. Install dependencies: pip install -r backend/requirements.txt")
        print("3. Run the application: python start_integrated.py")
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
