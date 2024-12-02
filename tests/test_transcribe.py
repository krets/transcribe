import json
import os
import sys
from unittest.mock import patch, mock_open, MagicMock
import pytest

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from transcribe import (
    read_file, extract_audio, summarize, transcribe_audio, get_transcription_for_file,
    find_input_file
)


MOCK_TRANSCRIPTION_OBJECT = {"segments": [{"start": 0, "text": "Hello"}]}
MOCK_TEST_TEXT = "Hello World!"

def test_read_file_json():
    with patch("builtins.open", mock_open(read_data=json.dumps(MOCK_TRANSCRIPTION_OBJECT))) as mock_file:
        transcription, text = read_file("test.json")
        assert transcription is not None
        assert text is None

def test_read_file_text():
    with patch("builtins.open", mock_open(read_data=MOCK_TEST_TEXT)) as mock_file:
        transcription, text = read_file("test.txt")
        assert transcription is None
        assert text == MOCK_TEST_TEXT


@patch("builtins.open", new_callable=mock_open)
def test_read_file_binary(mock_open):
    mock_file = mock_open.return_value.__enter__.return_value
    mock_file.read.side_effect = UnicodeDecodeError("utf-8", b'\x80\x81\x82', 0, 1, "invalid start byte")

    transcription, text = read_file("binary file")

    assert transcription is None
    assert text is None

@patch("os.path.exists", return_value=True)
@patch("os.remove")
@patch("subprocess.run")
def test_extract_audio(mock_run, mock_remove, mock_exists):
    mock_run.return_value.returncode = 0
    basename = "my_test_file"
    output_file = extract_audio(f"{basename}.mp4")
    assert output_file == f"{basename}.mp3"
    mock_remove.assert_called_once_with(f"{basename}.mp3")
    mock_run.assert_called_once()

@patch("requests.post")
@patch("builtins.open", new_callable=mock_open, read_data=b"dummy audio data")
def test_transcribe_audio(mock_open, mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = MOCK_TRANSCRIPTION_OBJECT
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    transcription = transcribe_audio("test.mp3")
    assert "segments" in transcription

@patch("transcribe.extract_audio")
@patch("transcribe.transcribe_audio")
@patch("os.path.exists", return_value=False)
@patch("os.path.getmtime", return_value=0)
def test_get_transcription_for_file(mock_getmtime, mock_exists, mock_transcribe_audio, mock_extract_audio):
    mock_transcribe_audio.return_value = MOCK_TRANSCRIPTION_OBJECT
    transcription = get_transcription_for_file("test.mp4", skip_cache=True)
    assert "segments" in transcription


@patch("transcribe.extract_audio")
@patch("transcribe.transcribe_audio")
@patch("os.path.exists", return_value=True)
@patch("os.path.getmtime", side_effect=[0, 1])
@patch("builtins.open", new_callable=mock_open, read_data=json.dumps(MOCK_TRANSCRIPTION_OBJECT))
def test_get_transcription_for_file_with_cache(mock_open, mock_getmtime, mock_exists, mock_transcribe_audio, mock_extract_audio):
    transcription = get_transcription_for_file("test.mp4", skip_cache=False)
    assert transcription == MOCK_TRANSCRIPTION_OBJECT
    mock_transcribe_audio.assert_not_called()

@patch("transcribe.extract_audio")
@patch("transcribe.transcribe_audio")
@patch("os.path.exists", return_value=True)
@patch("os.path.getmtime", side_effect=[2, 1])  # make cache file older than input_file
@patch("builtins.open", new_callable=mock_open, read_data=json.dumps(MOCK_TRANSCRIPTION_OBJECT))
def test_get_transcription_for_file_with_old_cache(mock_open, mock_getmtime, mock_exists, mock_transcribe_audio, mock_extract_audio):
    mock_transcribe_audio.return_value = {"segments": [{"start": 0, "text": "Hello"}]}
    transcription = get_transcription_for_file("test.mp4", skip_cache=False)
    assert "segments" in transcription
    mock_transcribe_audio.assert_called()

@patch("glob.glob", return_value=['file1.mp4', 'file2.mp4'])  # file2 is newest
@patch('os.path.getmtime', side_effect=[1, 2])
@patch('builtins.print')
def test_find_input_file_with_mp4_files(mock_print, mock_getmtime, mock_glob):
    result = find_input_file()
    assert result == 'file2.mp4'

@patch("glob.glob", return_value=[])
def test_find_input_file_no_mp4_files(mock_glob):
    with pytest.raises(FileNotFoundError):
        find_input_file()

@patch("requests.post")
def test_summarize(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        'choices': [{'message': {'content': MOCK_TEST_TEXT}}]
    }
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response
    summary = summarize("text")

    assert summary == MOCK_TEST_TEXT
    mock_post.assert_called_once()

@patch("requests.post")
def test_summarize_with_extra_prompt(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        'choices': [{'message': {'content': MOCK_TEST_TEXT}}]
    }
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response
    summary = summarize("text", "extra")

    assert summary == MOCK_TEST_TEXT
    mock_post.assert_called_once()