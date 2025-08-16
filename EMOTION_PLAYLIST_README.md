# Emotion-Based Playlist Feature

## Overview
The emotion detection system now includes a new playlist feature that creates personalized music playlists based on detected emotions. Instead of playing a single song, the system now generates a curated playlist of 6-8 songs that match the user's emotional state.

## New Features

### 1. Emotion-Based Playlist Generation
- **API Endpoint**: `POST /api/emotions/playlist`
- **Input**: Emotion detected from facial analysis
- **Output**: Curated playlist of 6-8 songs matching the emotion

### 2. Emotion-to-Music Mapping
The system maps each emotion to specific music genres and search terms:

- **Happy**: Upbeat songs, feel-good music, positive vibes, summer hits, dance music
- **Sad**: Melancholy songs, sad ballads, emotional music, heartbreak songs, reflective music
- **Angry**: Intense rock music, powerful songs, aggressive music, metal songs, energetic rock
- **Natural**: Calm relaxing music, ambient music, peaceful songs, nature sounds, meditation music
- **Disgust**: Dark intense music, heavy metal, industrial music, aggressive songs, intense electronic
- **Surprise**: Energetic exciting music, upbeat electronic, dance hits, party music, energetic pop

### 3. Enhanced User Experience
- **New Button**: "🎵 Play Emotion Playlist" replaces the single song button
- **Playlist Queue**: Automatically queues multiple tracks for continuous playback
- **Dashboard Integration**: Redirects to Spotify dashboard with playlist information
- **Smart Notifications**: Shows playlist size and first track information

## Technical Implementation

### Backend Changes
1. **New API Endpoint**: `/api/emotions/playlist`
2. **Multiple Search Queries**: Uses 5 different search terms per emotion
3. **Duplicate Removal**: Ensures no duplicate tracks in the playlist
4. **Queue Management**: Automatically queues additional tracks after the first

### Frontend Changes
1. **Updated UI**: New playlist button and messaging
2. **Enhanced Notifications**: Better display of playlist information
3. **Dashboard Integration**: Improved notification system for playlists

## Usage Flow

1. **Emotion Detection**: User analyzes their emotions using the camera
2. **Playlist Creation**: System generates a personalized playlist based on detected emotion
3. **Music Playback**: First track starts playing immediately
4. **Queue Management**: Remaining tracks are queued for continuous playback
5. **Dashboard Redirect**: User is redirected to Spotify dashboard with playlist info

## Benefits

- **Better Music Discovery**: Users get multiple song recommendations instead of one
- **Continuous Playback**: No need to manually select new songs
- **Emotion-Specific Curation**: Each emotion gets a carefully curated selection of music
- **Improved User Engagement**: Longer listening sessions with varied music

## API Response Format

```json
{
  "success": true,
  "emotion": "Happy",
  "playlist": [
    {
      "name": "Song Name",
      "artist": "Artist Name",
      "album": "Album Name",
      "cover_url": "https://...",
      "uri": "spotify:track:...",
      "preview_url": "https://..."
    }
  ],
  "total_tracks": 8,
  "first_track_playing": true,
  "message": "Created a happy mood playlist with 8 songs!"
}
```

## Future Enhancements

- **Playlist Saving**: Save emotion-based playlists to user's Spotify account
- **Mood Tracking**: Track emotional patterns over time
- **Personalization**: Learn user preferences for each emotion
- **Collaborative Playlists**: Share emotion-based playlists with friends
