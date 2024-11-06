#!/usr/bin/env python3
import argparse
import datetime
import logging
import os
import subprocess

import dotenv
import requests

dotenv.load_dotenv()

LOG = logging.getLogger('krets')
OPEN_AI_KEY = os.environ.get('OPENAI_API_KEY')

def extract_audio(video_file):
    output_file = os.path.splitext(os.path.basename(video_file))[0] + ".mp3"
    if os.path.exists(output_file):
        os.remove(output_file)

    command = f"ffmpeg -i {video_file} -codec:a libmp3lame -qscale:a 5 -ac 1 -ar 22050 {output_file}"

    LOG.info("Extracting audio: %s", command)
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        raise ChildProcessError(f"Error running command: {command}\n{result.stderr}")
    return output_file

def transcribe_audio(audio_file):
    LOG.info("Transcribing with whisper API")
    url = 'https://api.openai.com/v1/audio/transcriptions'
    headers = {
        'Authorization': f"Bearer {OPEN_AI_KEY}"
    }

    files = {
        'file': open(audio_file, 'rb')
    }

    data = {
        'model': 'whisper-1'
    }

    response = requests.post(url, headers=headers, files=files, data=data)
    response.raise_for_status()
    return response.json().get('text')

def summarize(text):
    LOG.info("Summarizing with GPT")
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Authorization': f"Bearer {OPEN_AI_KEY}"
    }
    data = {
        'model': 'gpt-4o-mini',
        'messages': [
            {
                'role': 'system',
                'content':
                    'This is a transcription summarizer. You will organize, and clarify the important points. '
                    'Translate everything to english. '
                    'Greetings and well-wishes are irrelevant. Do not include this information. '
                    'All output will be markdown. '
                    "Don't drop any important points. "
                    "Prefer unordered lists. "
                    "Use subheadings instead of bold or italic"
                    'The top-line header should be the date in Y-m-d format. '
            },
            {'role': 'user', 'content': text}
        ],
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    return result['choices'][0]['message']['content']


def main():
    parser = argparse.ArgumentParser(description="Transcribe and summarize audio from an FFmpeg-compatible file.")
    parser.add_argument("input_file", help="Path to text file or FFmpeg-compatible media file (e.g., .mp4, .mkv, .mov)")
    parser.add_argument("-t", "--transcription-only", action="store_true", help="Only output the transcription text.")
    args = parser.parse_args()

    input_file = args.input_file
    transcription_only = args.transcription_only

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read(1024)
            text += f.read()
            LOG.info("Input file appears to be plaintext.")
    except UnicodeDecodeError:
        text = None

    if text is None:
        audio_file = extract_audio(input_file)
        text = transcribe_audio(audio_file)

    base_name = os.path.basename(input_file)
    if base_name.startswith("202") and base_name[3].isdigit() and len(base_name) > 10:
        assumed_date = ''.join(base_name[:10])
    else:
        assumed_date = datetime.datetime.now().strftime("%Y-%m-%d")

    text = f"Date: {assumed_date}\n\n{text}"

    if transcription_only:
        print(f"Transcription:\n{text}")
    else:
        summary = summarize(text)
        print(f"Summary from chatGPT:\n{summary}\n")


if __name__ == '__main__':
    LOG.addHandler(logging.StreamHandler())
    LOG.handlers[-1].setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    LOG.setLevel(logging.DEBUG)
    main()
