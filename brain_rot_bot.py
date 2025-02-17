
from kokoro import KPipeline
from IPython.display import display, Audio
import numpy as np
import soundfile as sf
import logging
import polars as pl
import uuid
from moviepy import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip
import os
from moviepy.video.fx import Loop
from openai import OpenAI
from ollama import chat
from ollama import ChatResponse
import regex as re
# import subprocess
# # Attempt to activate the conda environment
# try:
#     # Construct the activation command based on the operating system
#     if os.name == 'nt':  # Windows
#         activate_command = 'conda activate ipynb_env_3.9'
#     else:  # macOS and Linux
#         activate_command = 'source activate ipynb_env_3.9'

#     subprocess.run(activate_command, shell=True, check=True, executable='/bin/bash')
#     print("Conda environment 'ipynb_env_3.9' activated successfully.")
# except subprocess.CalledProcessError as e:
#     print(f"Failed to activate conda environment: {e}")
# except FileNotFoundError:
#     print("Conda is not installed or not in your PATH.")
# except Exception as e:
#     print(f"An unexpected error occurred: {e}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Audio:
    def __init__(self, lang_code, voice='af_bella', rate=24000):
        self.pipeline = KPipeline(lang_code=lang_code)
        self.voice = voice
        self.rate = rate

    def compute_speed(self, text):
        """Compute a speed factor based on simple text cues."""
        if "!" in text or "?" in text:
            return 1.15
        if len(text.split()) > 20 or text.count(".") >= 3:
            return 0.9
        return 1.05

    def generate_audio_segments(self, text):
        """Generates audio segments and returns them as a list of NumPy arrays."""
        dynamic_speed = self.compute_speed(text)
        gen_dynamic = self.pipeline(
            text,
            voice=self.voice,
            speed=dynamic_speed,
            split_pattern=r'\n+'
        )

        audio_segments = []
        for i, (_, _, audio) in enumerate(gen_dynamic):
            logging.info(f"Generating audio segment {i}")
            audio_segments.append(audio)

        return audio_segments

    def merge_audio_segments(self, audio_segments):
        """Merges a list of audio segment arrays, writes to file, and returns file path."""
        merged_audio = np.concatenate(audio_segments)
        filename = f"{uuid.uuid4()}.wav"
        sf.write(filename, merged_audio, self.rate)
        logging.info(f"Merged audio file created directly from segments in memory: [FILE NAME] {filename}")
        return filename

    def generate_audio(self, text):
        """Generates audio segments and merges them in memory to create the final audio."""
        audio_segments = self.generate_audio_segments(text)
        audio_file_path = self.merge_audio_segments(audio_segments)
        return audio_file_path

class Video:
    def create_synchronized_text(self, audio_path, text_content, font_path="roboto/Roboto-Black.ttf"):
        video = VideoFileClip("outputs/clip.mp4").with_volume_scaled(0.0)
        audio_clip = AudioFileClip(audio_path)

        # Adjust video duration: loop if shorter than audio, or trim if longer.
        if video.duration < audio_clip.duration:
            loop_effect = Loop(duration=audio_clip.duration).copy()
            video = loop_effect.apply(video)
        else:
            video = video.subclipped(0, audio_clip.duration)

        video = video.with_audio(audio_clip)
        video.with_volume_scaled(0.8)

        words = text_content.split()
        word_duration = audio_clip.duration / len(words)

        text_clips = []
        for i, word in enumerate(words):
            txt = (TextClip(text=word, font=font_path, font_size=70, color='white', method="caption", size=(1920, 1080))
                   .with_duration(word_duration)
                   .with_start(i * word_duration)
                   .with_position('center'))
            text_clips.append(txt)

        return CompositeVideoClip([video] + text_clips)

class LLM:
    def __init__(self, api_key, base_url, perplexity_model):
        self.api_key = api_key
        self.base_url = base_url
        self.perplexity_model = perplexity_model
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_video_title(self, response):
        ollama_response: ChatResponse = chat(model='llama3.2:latest', messages=[
        {
            'role': 'system',
            'content': (
                "You are a youtube shorts creator, the user will provide you with a script."
                "Your view point should be in the form of a naive female college student with no prior knowledge of the topic."
                "Your task is to come up with a quirky short title for the script, Only return the title to the user, nothing else."
            )
        },
        {
            'role': 'user',
            'content': response,
        },
        ])
        return ollama_response.message.content

    def generate_llm_response(self, transcript):
        function_description = '''
        def compute_speed(text):
            """Compute a speed factor based on simple text cues."""
            if "!" in text or "?" in text:
                return 1.15
            # Slow down if text is long or contains many commas
            if len(text.split()) > 20 or text.count(",") >= 3:
                return 0.9
            return 1.05
        '''

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a youtube shorts creator, the user will provide you with a basic idea."
                    "You are tasked with creating an engaging and enjoyable short narration transcript. You should never describe the scene, just the dialogue."
                    "You should fact check every bit of your script as we can not let our viewers down, but your response should never leave in [n] or anything describing the speed."
                    "Your view point should be in the form of a naive female college student with no prior knowledge of the topic."
                   
                ),
            },
            {
                "role": "user",
                "content": transcript,
            },
        ]

        response = self.client.chat.completions.create(
            model= self.perplexity_model,
            messages=messages,
        )

        return response.choices[0].message.content

    def format_title_for_mp4(self, title):
        # Remove characters that may be invalid in file systems
        sanitized = re.sub(r'[\\/*?:"<>|]', '', title)
        # Replace spaces with underscores for a cleaner file name
        sanitized = sanitized.replace(" ", "_")
        # Optionally, limit the length of the filename
        return sanitized[:100]


def main():
    lang_code = 'a'
    data_file = pl.read_csv('hf://datasets/SocialGrep/one-million-reddit-confessions/one-million-reddit-confessions.csv')
    audio_generator = Audio(lang_code)
    video_generator = Video()
    api_key = '####'
    base_url = "https://api.perplexity.ai"
    perplexity_model = 'sonar'
    llm_generator = LLM(api_key, base_url, perplexity_model)

    max_attempts = 50
    attempt = 0

    while attempt < max_attempts:
        rand_row = data_file.select(["selftext", "title"]).sample(n=1)
        selftext_text = rand_row["selftext"][0]
        title_text = rand_row["title"][0]
        if selftext_text != "[removed]" and selftext_text != "[deleted]":
            combined_text = f"{title_text}\n{selftext_text}"

            # llm_response = llm_generator.generate_llm_response(combined_text) - This sends an actual perplexity request, we shouldnt use this.
            llm_response = combined_text
            logging.info(f"LLM Response: {llm_response}")
            llm_title_generator = llm_generator.generate_video_title(combined_text)
            logging.info(f"Generated Video Title: {llm_title_generator}")

            audio_file_path = audio_generator.generate_audio(llm_response) # Use LLM response for audio
            final_clip = video_generator.create_synchronized_text(audio_file_path, llm_response) # Use LLM response for text overlay
            title = llm_generator.format_title_for_mp4(llm_title_generator)
            final_clip.write_videofile(
                f"{title}.mp4",
                fps=24,
                codec="libx264",
                threads=os.cpu_count(),
                preset="ultrafast"
            )
            os.remove(audio_file_path)
            break
        attempt += 1
    else:
        raise ValueError("No valid entry found in dataset after several attempts.")

if __name__ == "__main__":
    main()