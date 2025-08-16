import spotipy
from spotipy.oauth2 import SpotifyOAuth

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="b52e7240c7544a589e65126efac853dc",
    client_secret="d54a64dfe1e144c4b2e0fa3cb256ad53",
    redirect_uri="http://127.0.0.1:8888/callback",
    scope="user-modify-playback-state user-read-playback-state user-read-currently-playing"
))

print("🎵 Testing Spotify API Connection...")

try:
    user = sp.current_user()
    print(f"✅ Connected as: {user['display_name']}")
    
    results = sp.search(q='test song', type='track', limit=1)
    if results['tracks']['items']:
        song = results['tracks']['items'][0]
        print(f"✅ Search works! Found: {song['name']} by {song['artists'][0]['name']}")
    
    current = sp.current_playback()
    if current and current['is_playing']:
        print(f"✅ Currently playing: {current['item']['name']}")
    else:
        print("ℹ️ No music currently playing (start Spotify and play something)")
    
    print("\n🎮 Testing playback controls (requires Spotify Premium)...")
    
    devices = sp.devices()
    if devices['devices']:
        print(f"✅ Found {len(devices['devices'])} device(s)")
        for device in devices['devices']:
            print(f"   - {device['name']} ({device['type']})")
    else:
        print("⚠️ No active devices found. Open Spotify app on your computer/phone")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\n🔧 Troubleshooting:")
    print("1. Check your CLIENT_ID and CLIENT_SECRET")
    print("2. Make sure redirect URI matches dashboard exactly")
    print("3. Open Spotify app and start playing music")

print("\n�� Test complete!")
