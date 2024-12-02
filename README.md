## Video Transcriber and Summary
Summarize a given audio/video file. This is mostly for generating meeting notes.

### Usage

    python transcribe.py [options] <audio/video file/JSON/Text>

#### Options:
    -t, --transcription-only   Output only the transcription text.
    -f, --force                Force re-caching of transcription.
    -p, --prompt <text>        Extra prompt to add to the summary directive.


### Requirements 
 - `OPENAI_API_KEY` environment variable
   - Whisper online is used for faster transcriptions
   - ChatGPT is used for Summary
 - `ffmpeg` is used to make a compressed, mono MP3 to send to OpenAI
