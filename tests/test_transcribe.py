import os
import sys
from unittest.mock import patch, mock_open, MagicMock

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from transcribe import read_file, extract_audio, transcribe_audio, get_transcription_for_file


def test_read_file_json():
    with patch("builtins.open", mock_open(read_data='{"segments": [{"start": 0, "text": "Hello"}]}')) as mock_file:
        transcription, text = read_file("test.json")
        assert transcription is not None
        assert text is None

def test_read_file_text():
    with patch("builtins.open", mock_open(read_data="Hello World")) as mock_file:
        transcription, text = read_file("test.txt")
        assert transcription is None
        assert text == "Hello World"

@patch("os.path.exists", return_value=True)
@patch("os.remove")
@patch("subprocess.run")
def test_extract_audio(mock_run, mock_remove, mock_exists):
    mock_run.return_value.returncode = 0
    output_file = extract_audio("test.mp4")
    assert output_file == "test.mp3"
    mock_remove.assert_called_once_with("test.mp3")
    mock_run.assert_called_once()

@patch("requests.post")
@patch("builtins.open", new_callable=mock_open, read_data=b"dummy audio data")
def test_transcribe_audio(mock_open, mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {"segments": [{"start": 0, "text": "Hello"}]}
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    transcription = transcribe_audio("test.mp3")
    assert "segments" in transcription

@patch("transcribe.extract_audio")
@patch("transcribe.transcribe_audio")
@patch("os.path.exists", return_value=False)
def test_get_transcription_for_file(mock_exists, mock_transcribe_audio, mock_extract_audio):
    mock_transcribe_audio.return_value = {"segments": [{"start": 0, "text": "Hello"}]}
    transcription = get_transcription_for_file("test.mp4", skip_cache=True)
    assert "segments" in transcription