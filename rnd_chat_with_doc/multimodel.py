import os
from pytube import YouTube
from pathlib import Path
from moviepy.editor import VideoFileClip
import speech_recognition as sr

# SET CONFIG
video_url = "https://www.youtube.com/watch?v=d_qvLDhkg00"
output_video_path = "./video_data/"
output_audio_path = "./audio_data/"
output_folder = "./mixed_data/"

filepath = output_video_path + "a.mp4"
audiopath = output_audio_path + "output_audio.wav"

# Create necessary directories
Path(output_video_path).mkdir(parents=True, exist_ok=True)  # Ensures video path exists
Path(output_audio_path).mkdir(parents=True, exist_ok=True)  # Ensures video path exists
Path(output_folder).mkdir(
    parents=True, exist_ok=True
)  # Ensures mixed data folder exists


def download_video(url, output_path):
    """
    Download a video from a given url and save it to the output path.

    Parameters:
    url (str): The url of the video to download.
    output_path (str): The path to save the video to.

    Returns:
    dict: A dictionary containing the metadata of the video.
    """
    try:
        yt = YouTube(url, use_oauth=True, allow_oauth_cache=True)
        metadata = {"Author": yt.author, "Title": yt.title, "Views": yt.views}
        yt.streams.get_highest_resolution().download(
            output_path=output_path, filename="input_vid.mp4"
        )
        return metadata
    except Exception as e:
        print(f"An error occurred while downloading the video: {e}")
        return None


def video_to_images(video_path, output_folder):
    """
    Convert a video to a sequence of images and save them to the output folder.

    Parameters:
    video_path (str): The path to the video file.
    output_folder (str): The path to the folder to save the images to.

    """
    clip = VideoFileClip(video_path)
    clip.write_images_sequence(
        os.path.join(output_folder, "frame%04d.png"),
        fps=1,  # configure this for controlling frame rate.
    )


def video_to_audio(video_path, output_audio_path):
    """
    Convert a video to audio and save it to the output path.

    Parameters:
    video_path (str): The path to the video file.
    output_audio_path (str): The path to save the audio to.

    """
    clip = VideoFileClip(video_path)
    audio = clip.audio
    audio.write_audiofile(output_audio_path)


def audio_to_text(audio_path):
    """
    Convert audio to text using the SpeechRecognition library.

    Parameters:
    audio_path (str): The path to the audio file.

    Returns:
    test (str): The text recognized from the audio.

    """
    recognizer = sr.Recognizer()
    audio = sr.AudioFile(audio_path)

    with audio as source:
        # Record the audio data
        audio_data = recognizer.record(source)

        try:
            # Recognize the speech
            text = recognizer.recognize_whisper(audio_data)
        except sr.UnknownValueError:
            print("Speech recognition could not understand the audio.")
        except sr.RequestError as e:
            print(f"Could not request results from service; {e}")

    return text


if __name__ == "__main__":
    # video_to_images(filepath, output_folder)
    # video_to_audio(filepath, audiopath)
    txt = audio_to_text(audiopath)
    print(txt)
