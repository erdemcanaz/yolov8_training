import yt_dlp
from pathlib import Path

URLS = [

]

def download_video(url):
    ydl_opts = {'format': 'best'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

if __name__ == "__main__":
    video_url = input("Enter YouTube video URL: ")
    if video_url == "":           
        for url in URLS:
           download_video(url)
    else:
        download_video(video_url)
 
