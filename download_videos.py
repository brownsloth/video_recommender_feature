import yt_dlp
import os

# --- CONFIGURATION ---
channel_url = "https://www.youtube.com/@veritasium/shorts"  # <<< can be @handle or full channel URL
output_dir = "videos"
max_videos = 50
os.makedirs(output_dir, exist_ok=True)

# Step 1: Configure yt-dlp options
ydl_opts = {
    'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),  # output file template
    'format': 'bestvideo+bestaudio/best',
    'noplaylist': False,  # still treat channels correctly
    'playlistend': max_videos,  # Download only first N videos
    'quiet': False,  # Show progress
    'match_filter': yt_dlp.utils.match_filter_func("duration < 60")
}

# Step 2: Download videos
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([channel_url])
