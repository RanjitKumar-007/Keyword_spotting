import os
import uuid
import tempfile
import logging
from pydub import AudioSegment

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"} # Audio formats

"""
Silero VAD (used internally by Faster-Whisper) requires audio whose sample
count is a multiple of this chunk size (30 ms × 16 000 Hz = 480 samples). 
"""

_VAD_CHUNK_MS = 30  # Should be in 10, 20, or 30 ms

class AudioPreprocessorError(Exception):
    pass

class AudioPreprocessor:
    """
    Converts any supported audio file to a 16kHz mono WAV whose length is
    padded to an exact multiple of VAD_CHUNK_MS to avoid Silero VAD's.
    """
    def __init__(self, target_sample_rate: int = 16000, target_channels: int = 1):
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels

    @staticmethod
    def _pad_to_vad_boundary(audio: AudioSegment, chunk_ms: int = _VAD_CHUNK_MS) -> AudioSegment:
        """
        Append silence so that the total duration is a multiple of chunk_ms.
        Silero VAD processes audio in fixed 10/20/30 ms frames.
        """
        remainder = len(audio) % chunk_ms
        if remainder:
            pad_ms = chunk_ms - remainder
            silence = AudioSegment.silent(duration=pad_ms, frame_rate=audio.frame_rate)
            audio = audio + silence
            logger.debug(f"Padded audio by {pad_ms} ms to align with {chunk_ms} ms VAD chunks.")
        return audio

    def preprocess(self, input_file: str, output_dir: str | None = None) -> str:
        if not os.path.exists(input_file):
            raise AudioPreprocessorError(f"Input file not found: {input_file}")
        ext = os.path.splitext(input_file)[-1].lower()
        if ext not in SUPPORTED_FORMATS:
            raise AudioPreprocessorError(
                f"Unsupported format '{ext}'. Supported: {SUPPORTED_FORMATS}"
            )
        out_dir = output_dir or tempfile.gettempdir()
        os.makedirs(out_dir, exist_ok=True)
        unique_name = f"preprocessed_{uuid.uuid4().hex}.wav"
        output_file = os.path.join(out_dir, unique_name)
        try:
            logger.info(f"Loading audio: {input_file}")
            audio = AudioSegment.from_file(input_file)
            logger.info(
                f"Original — channels: {audio.channels}, "
                f"sample rate: {audio.frame_rate} Hz, "
                f"duration: {len(audio) / 1000:.2f}s"
            )
            audio = audio.set_channels(self.target_channels)
            audio = audio.set_frame_rate(self.target_sample_rate)
            audio = self._pad_to_vad_boundary(audio)  # fix FRAME_DURATION_MS warning
            audio.export(output_file, format="wav")
            logger.info(f"Preprocessed audio saved: {output_file}")
            return output_file
        except AudioPreprocessorError:
            raise
        except Exception as e:
            raise AudioPreprocessorError(f"Preprocessing failed: {e}") from e
    @staticmethod
    def cleanup(file_path: str):
        """Remove temp WAV after pipeline completes."""
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Temp file removed: {file_path}")
        except OSError as e:
            logger.warning(f"Could not remove temp file {file_path}: {e}")
            