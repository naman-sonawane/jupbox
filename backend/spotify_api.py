from flask import Flask, request, jsonify
from flask_cors import CORS
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

SPOTIFY_CLIENT_ID = "b52e7240c7544a589e65126efac853dc"
SPOTIFY_CLIENT_SECRET = "d54a64dfe1e144c4b2e0fa3cb256ad53"
REDIRECT_URI = "http://127.0.0.1:8888/callback"

sp = None

def get_spotify_client():
    global sp
    if sp is None:
        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET,
            redirect_uri=REDIRECT_URI,
            scope="user-modify-playback-state user-read-playback-state user-read-currently-playing user-read-private"
        ))
    return sp

@app.route('/api/spotify/current', methods=['GET'])
def get_current_track():
    try:
        sp = get_spotify_client()
        current = sp.current_playback()
        
        if current and current['item']:
            track = current['item']
            return jsonify({
                'success': True,
                'track': {
                    'name': track['name'],
                    'artist': track['artists'][0]['name'],
                    'album': track['album']['name'],
                    'cover_url': track['album']['images'][0]['url'] if track['album']['images'] else None,
                    'is_playing': current['is_playing'],
                    'progress_ms': current['progress_ms'],
                    'duration_ms': track['duration_ms'],
                    'uri': track['uri']
                }
            })
        else:
            return jsonify({
                'success': True,
                'track': None,
                'message': 'No track currently playing'
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/spotify/play', methods=['POST'])
def play_track():
    try:
        sp = get_spotify_client()
        current = sp.current_playback()
        
        if current and current['is_playing']:
            sp.pause_playback()
            action = 'paused'
        else:
            sp.start_playback()
            action = 'playing'
            
        return jsonify({'success': True, 'action': action})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/spotify/next', methods=['POST'])
def next_track():
    try:
        sp = get_spotify_client()
        sp.next_track()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/spotify/previous', methods=['POST'])
def previous_track():
    try:
        sp = get_spotify_client()
        sp.previous_track()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/spotify/user', methods=['GET'])
def get_user_info():
    try:
        sp = get_spotify_client()
        user = sp.current_user()
        return jsonify({
            'success': True,
            'user': {
                'name': user['display_name'],
                'id': user['id'],
                'image': user['images'][0]['url'] if user['images'] else None
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/spotify/devices', methods=['GET'])
def get_devices():
    try:
        sp = get_spotify_client()
        devices = sp.devices()
        return jsonify({
            'success': True,
            'devices': devices['devices']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/spotify/search', methods=['GET'])
def search_tracks():
    query = request.args.get('q', '')
    if not query:
        return jsonify({'success': False, 'error': 'Query parameter required'}), 400
    
    try:
        sp = get_spotify_client()
        results = sp.search(q=query, type='track', limit=10)
        tracks = []
        
        for track in results['tracks']['items']:
            tracks.append({
                'name': track['name'],
                'artist': track['artists'][0]['name'],
                'album': track['album']['name'],
                'cover_url': track['album']['images'][0]['url'] if track['album']['images'] else None,
                'uri': track['uri']
            })
            
        return jsonify({'success': True, 'tracks': tracks})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/spotify/play-track', methods=['POST'])
def play_specific_track():
    data = request.get_json()
    track_uri = data.get('uri')
    
    if not track_uri:
        return jsonify({'success': False, 'error': 'Track URI required'}), 400
    
    try:
        sp = get_spotify_client()
        sp.start_playback(uris=[track_uri])
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("🎵 Starting Spotify API server...")
    print("📡 Server will be available at http://localhost:5000")
    print("🔗 Make sure to run the Spotify OAuth flow first!")
    app.run(debug=True, port=5000)
