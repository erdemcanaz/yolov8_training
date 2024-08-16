import yt_dlp
from pathlib import Path
import os

#=========================================================================================================
URLS = [

]
#=========================================================================================================

EXPORT_FOLDER_PATH = Path(__file__).resolve().parent.parent / "exports"

def delete_files_in_folder(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Iterate through each file
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            # Check if it's a file and delete it
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

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
 
