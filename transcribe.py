#!/usr/bin/env python3
import argparse
import glob
import json
import logging
import os
import subprocess
from datetime import datetime, timedelta

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
        'model': 'whisper-1',
        'response_format': 'verbose_json'
    }

    response = requests.post(url, headers=headers, files=files, data=data)
    response.raise_for_status()
    return response.json()

def summarize(text, extra_prompt=None):
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
                    "Use subheadings instead of bold or italic. "
                    'The top-line header should be the date in Y-m-d format. '
            },
            {'role': 'user', 'content': text}
        ],
    }
    if extra_prompt:
        data['messages'].append({'role': 'system', 'content': extra_prompt})
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    return result['choices'][0]['message']['content']


def get_transcription_for_file(input_file, skip_cache=False):
    basename = os.path.basename(input_file)
    dirname = os.path.dirname(input_file)
    base_basename = os.path.splitext(basename)[0]
    cached_transcription = os.path.join(dirname, f".{base_basename}.json")
    cache_mtime = os.path.getmtime(cached_transcription) if os.path.exists(cached_transcription) else 0

    if skip_cache or (cache_mtime < os.path.getmtime(input_file)):
        audio_file = extract_audio(input_file)
        transcription = transcribe_audio(audio_file)
        LOG.debug("Caching transcription to: %s", cached_transcription)
        with open(cached_transcription, 'w') as fh:
            json.dump(transcription, fh)
    else:
        LOG.info("Loading cached transcription from: %s", cached_transcription)
        with open(cached_transcription, 'r') as fh:
            transcription = json.load(fh)
    return transcription


def main():
    parser = argparse.ArgumentParser(description="Transcribe and summarize audio from an FFmpeg-compatible file.")
    parser.add_argument("input_file", nargs="?", help="Path to text file, JSON, or FFmpeg-compatible media file (e.g., .mp4, .mkv, .mov)")
    parser.add_argument("-t", "--transcription-only", action="store_true", help="Only output the transcription text.")
    parser.add_argument("-f", "--force", action="store_true", help="Force re-caching transcription")
    parser.add_argument("-p", "--prompt", default="", help="Extra prompt to add to the summary directive.")
    args = parser.parse_args()

    input_file = args.input_file
    transcription = None
    text = None

    # If no input file is specified, select the latest .mp4 file in the current directory
    if not input_file:
        mp4_files = sorted(glob.glob("*.mp4"), key=os.path.getmtime, reverse=True)
        if mp4_files:
            input_file = mp4_files[0]
            print(f"No input_file specified. Using the latest .mp4 file: {input_file}")
        else:
            raise FileNotFoundError("No .mp4 files found in the current directory.")

    base_name = os.path.basename(input_file)

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            if input_file.endswith('.json'):
                transcription = json.load(f)
                LOG.info("Input file appears to be a JSON file.")
            else:
                text = f.read(1024)
                text += f.read()
                LOG.info("Input file appears to be plaintext.")
    except UnicodeDecodeError:
        text = None

    if text is None and transcription is None:
        transcription = get_transcription_for_file(input_file, skip_cache=args.force)

    if base_name.startswith("20") and base_name[:4].isdigit() and len(base_name) > 10:
        # OBS saves files with title like: %Y-%m-%d_%H-%M-%S
        assumed_date = ''.join(base_name[:10])
    else:
        assumed_date = datetime.fromtimestamp(os.path.getmtime(input_file)).strftime('%Y-%m-%d')

    if transcription:
        text = f"Filename: {base_name}\nDate: {assumed_date}\n\n"
        for seg in transcription['segments']:
            text += f"[{timedelta(seconds=seg['start'])}] {seg['text']}\n"

    if args.transcription_only:
        print(f"Transcription:\n{text}")
    else:
        summary = summarize(text, args.prompt)
        print(f"Summary from chatGPT:\n{summary}\n")


if __name__ == '__main__':
    LOG.addHandler(logging.StreamHandler())
    LOG.handlers[-1].setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    LOG.setLevel(logging.DEBUG)
    main()
