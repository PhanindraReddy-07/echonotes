"""
Transcriber - Offline Speech Recognition using Vosk
====================================================
Converts audio to text with word-level timestamps and confidence scores
"""
import json
import wave
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Generator, Union
from dataclasses import dataclass, field

import numpy as np

# Lazy import for Vosk
def _get_vosk():
    try:
        import vosk
        vosk.SetLogLevel(-1)  # Suppress Vosk logs
        return vosk
    except ImportError:
        raise ImportError(
            "Vosk is required for transcription. Install with: pip install vosk\n"
            "Also download a model from: https://alphacephei.com/vosk/models"
        )


@dataclass
class Word:
    """Represents a single transcribed word"""
    text: str
    start_time: float  # seconds
    end_time: float    # seconds
    confidence: float  # 0.0 to 1.0
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict:
        return {
            'text': self.text,
            'start': round(self.start_time, 3),
            'end': round(self.end_time, 3),
            'confidence': round(self.confidence, 3)
        }


@dataclass
class Utterance:
    """Represents a continuous speech segment (sentence/phrase)"""
    text: str
    words: List[Word]
    start_time: float
    end_time: float
    speaker: Optional[str] = None  # Will be filled by diarizer
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def avg_confidence(self) -> float:
        if not self.words:
            return 0.0
        return sum(w.confidence for w in self.words) / len(self.words)
    
    @property
    def word_count(self) -> int:
        return len(self.words)
    
    def to_dict(self) -> Dict:
        return {
            'text': self.text,
            'start': round(self.start_time, 3),
            'end': round(self.end_time, 3),
            'speaker': self.speaker,
            'confidence': round(self.avg_confidence, 3),
            'words': [w.to_dict() for w in self.words]
        }


@dataclass 
class TranscriptionResult:
    """Complete transcription result"""
    text: str                          # Full transcript text
    utterances: List[Utterance]        # List of utterances
    words: List[Word]                  # All words with timestamps
    duration: float                    # Total audio duration
    language: str = "en"               # Detected/specified language
    model_name: str = ""               # Model used for transcription
    
    @property
    def word_count(self) -> int:
        return len(self.words)
    
    @property
    def avg_confidence(self) -> float:
        if not self.words:
            return 0.0
        return sum(w.confidence for w in self.words) / len(self.words)
    
    @property
    def low_confidence_words(self) -> List[Word]:
        """Words with confidence below 0.7"""
        return [w for w in self.words if w.confidence < 0.7]
    
    def get_text_with_timestamps(self, interval: float = 30.0) -> str:
        """Get text with timestamp markers every N seconds"""
        lines = []
        current_time = 0.0
        current_text = []
        
        for word in self.words:
            if word.start_time >= current_time + interval:
                if current_text:
                    timestamp = f"[{self._format_time(current_time)}]"
                    lines.append(f"{timestamp} {' '.join(current_text)}")
                current_text = []
                current_time = word.start_time
            current_text.append(word.text)
        
        # Add remaining text
        if current_text:
            timestamp = f"[{self._format_time(current_time)}]"
            lines.append(f"{timestamp} {' '.join(current_text)}")
        
        return '\n'.join(lines)
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS or HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"
    
    def to_dict(self) -> Dict:
        return {
            'text': self.text,
            'duration': round(self.duration, 2),
            'word_count': self.word_count,
            'avg_confidence': round(self.avg_confidence, 3),
            'language': self.language,
            'model': self.model_name,
            'utterances': [u.to_dict() for u in self.utterances]
        }


class Transcriber:
    """
    Offline Speech-to-Text using Vosk
    
    Features:
    - Completely offline (no internet required)
    - Word-level timestamps
    - Confidence scores
    - Multiple language support
    - Streaming transcription
    
    Usage:
        transcriber = Transcriber(model_path="path/to/vosk-model")
        result = transcriber.transcribe(audio_data)
        print(result.text)
    """
    
    # Model download URLs
    MODEL_URLS = {
        'en-us': 'https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip',
        'en-us-lgraph': 'https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip',
        'en-us-small': 'https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip',
        'en-in': 'https://alphacephei.com/vosk/models/vosk-model-en-in-0.5.zip',
        'en-in-small': 'https://alphacephei.com/vosk/models/vosk-model-small-en-in-0.4.zip',
        'hi': 'https://alphacephei.com/vosk/models/vosk-model-hi-0.22.zip',
        'hi-small': 'https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip',
        'te-small': 'https://alphacephei.com/vosk/models/vosk-model-small-te-0.42.zip',
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "vosk-model-en-us-0.22",
        sample_rate: int = 16000
    ):
        """
        Initialize the transcriber
        
        Args:
            model_path: Path to Vosk model directory
            model_name: Name of the model (for reference)
            sample_rate: Expected audio sample rate
        """
        self.vosk = _get_vosk()
        self.sample_rate = sample_rate
        self.model_name = model_name
        self.model = None
        self._model_path = model_path
        
        if model_path:
            self._load_model(model_path)
    
    def _load_model(self, model_path: str) -> None:
        """Load Vosk model from path"""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}\n"
                f"Download from: https://alphacephei.com/vosk/models"
            )
        
        print(f"[Transcriber] Loading model from {model_path}...")
        self.model = self.vosk.Model(str(model_path))
        self._model_path = str(model_path)
        print(f"[Transcriber] Model loaded successfully")
    
    def set_model(self, model_path: str) -> None:
        """Set/change the model"""
        self._load_model(model_path)
    
    @classmethod
    def get_or_download_model(cls, model_key: str = 'en-us-small') -> str:
        """
        Get model path, downloading if necessary.
        
        Checks in order:
        1. Project's models/ folder
        2. User cache folder
        3. Downloads if not found
        
        Args:
            model_key: Key from MODEL_URLS (default: 'en-us-small' ~50MB)
            
        Returns:
            Path to the model directory
        """
        import urllib.request
        import zipfile
        import os
        
        # Model name mapping
        model_name_map = {
            'en-us': 'vosk-model-en-us-0.22',
            'en-us-small': 'vosk-model-small-en-us-0.15',
            'en-in': 'vosk-model-en-in-0.5',
            'en-in-small': 'vosk-model-small-en-in-0.4',
            'hi': 'vosk-model-hi-0.22',
            'hi-small': 'vosk-model-small-hi-0.22',
        }
        
        model_name = model_name_map.get(model_key, 'vosk-model-small-en-us-0.15')
        
        # Search locations in order of priority
        search_paths = [
            # 1. Project's models/ folder (relative to this file)
            Path(__file__).parent.parent / "models",
            # 2. Current working directory models/
            Path.cwd() / "models",
            # 3. echonotes/models/ 
            Path.cwd() / "echonotes" / "models",
            # 4. User cache
            Path.home() / ".cache" / "echonotes" / "vosk-models",
        ]
        
        # Check each location for any vosk model
        for search_dir in search_paths:
            if not search_dir.exists():
                continue
            
            # Look for exact model name
            exact_path = search_dir / model_name
            if exact_path.exists():
                print(f"[Transcriber] Found model at {exact_path}")
                return str(exact_path)
            
            # Look for any vosk model in the directory
            for item in search_dir.iterdir():
                if item.is_dir() and 'vosk-model' in item.name:
                    print(f"[Transcriber] Found model at {item}")
                    return str(item)
        
        # Not found - download to cache
        cache_dir = Path.home() / ".cache" / "echonotes" / "vosk-models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / model_name
        
        if model_key not in cls.MODEL_URLS:
            model_key = 'en-us-small'
        
        url = cls.MODEL_URLS[model_key]
        zip_path = cache_dir / f"{model_name}.zip"
        
        print(f"[Transcriber] No model found. Downloading from {url}...")
        print(f"[Transcriber] This may take a few minutes (~50MB)...")
        
        try:
            urllib.request.urlretrieve(url, str(zip_path))
            print(f"[Transcriber] Download complete. Extracting...")
            
            with zipfile.ZipFile(str(zip_path), 'r') as zip_ref:
                zip_ref.extractall(str(cache_dir))
            
            os.remove(str(zip_path))
            
            print(f"[Transcriber] Model ready at {model_path}")
            return str(model_path)
            
        except Exception as e:
            print(f"[Transcriber] Failed to download model: {e}")
            print(f"[Transcriber] Please download manually from: https://alphacephei.com/vosk/models")
            raise
    
    def transcribe(
        self,
        audio_input,  # Can be AudioData object OR file path string
        show_progress: bool = True
    ) -> TranscriptionResult:
        """
        Transcribe audio to text
        
        Args:
            audio_input: AudioData object OR path to audio file (str/Path)
            show_progress: Show progress during transcription
            
        Returns:
            TranscriptionResult with full transcript and word details
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call set_model() first or provide model_path in constructor.")
        
        # Handle file path input
        if isinstance(audio_input, (str, Path)):
            audio_data = self._load_audio_file(str(audio_input))
        else:
            audio_data = audio_input
        
        samples = audio_data.samples
        sr = audio_data.sample_rate
        
        # Ensure correct sample rate
        if sr != self.sample_rate:
            samples = self._resample(samples, sr, self.sample_rate)
        
        # Convert to 16-bit PCM
        audio_bytes = self._to_pcm_bytes(samples)
        
        # Create recognizer
        recognizer = self.vosk.KaldiRecognizer(self.model, self.sample_rate)
        recognizer.SetWords(True)
        
        # Process audio
        all_words = []
        utterances = []
        
        chunk_size = 4000
        total_chunks = len(audio_bytes) // chunk_size + 1
        
        if show_progress:
            print(f"[Transcriber] Processing {audio_data.duration:.1f}s of audio...")
        
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            
            if recognizer.AcceptWaveform(chunk):
                result = json.loads(recognizer.Result())
                words, utterance = self._parse_result(result)
                all_words.extend(words)
                if utterance:
                    utterances.append(utterance)
            
            if show_progress and i % (chunk_size * 50) == 0:
                progress = min(100, int(i / len(audio_bytes) * 100))
                print(f"[Transcriber] Progress: {progress}%", end='\r')
        
        # Get final result
        final_result = json.loads(recognizer.FinalResult())
        words, utterance = self._parse_result(final_result)
        all_words.extend(words)
        if utterance:
            utterances.append(utterance)
        
        if show_progress:
            print(f"[Transcriber] Progress: 100% - Complete!")
        
        # Build full text
        full_text = ' '.join(w.text for w in all_words)
        
        return TranscriptionResult(
            text=full_text,
            utterances=utterances,
            words=all_words,
            duration=audio_data.duration,
            language="en",
            model_name=self.model_name
        )
    
    def _load_audio_file(self, filepath: str):
        """
        Load audio file and return AudioData-like object
        
        Args:
            filepath: Path to audio file (wav, mp3, etc.)
            
        Returns:
            Object with samples, sample_rate, duration attributes
        """
        from dataclasses import dataclass
        import wave
        
        @dataclass
        class SimpleAudioData:
            samples: np.ndarray
            sample_rate: int
            duration: float
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath}")
        
        # Handle WAV files
        if filepath.suffix.lower() == '.wav':
            with wave.open(str(filepath), 'rb') as wf:
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                
                # Read raw bytes
                raw_data = wf.readframes(n_frames)
                
                # Convert to numpy array
                if sample_width == 2:
                    dtype = np.int16
                elif sample_width == 4:
                    dtype = np.int32
                else:
                    dtype = np.int16
                
                samples = np.frombuffer(raw_data, dtype=dtype)
                
                # Convert to mono if stereo
                if n_channels == 2:
                    samples = samples.reshape(-1, 2).mean(axis=1)
                
                # Convert to float32 normalized
                samples = samples.astype(np.float32) / np.iinfo(dtype).max
                
                duration = n_frames / sample_rate
                
                return SimpleAudioData(
                    samples=samples,
                    sample_rate=sample_rate,
                    duration=duration
                )
        
        # Handle other formats using pydub (preferred) or ffmpeg
        else:
            import tempfile
            
            # Try pydub first (handles many formats without external ffmpeg)
            try:
                from pydub import AudioSegment
                
                # Load with pydub
                audio = AudioSegment.from_file(str(filepath))
                
                # Convert to mono, 16kHz
                audio = audio.set_channels(1).set_frame_rate(16000)
                
                # Export to temporary WAV
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp_path = tmp.name
                
                try:
                    audio.export(tmp_path, format='wav')
                    return self._load_audio_file(tmp_path)
                finally:
                    if Path(tmp_path).exists():
                        Path(tmp_path).unlink()
                        
            except ImportError:
                pass  # pydub not available, try ffmpeg
            except Exception as pydub_error:
                print(f"[Transcriber] pydub failed: {pydub_error}, trying ffmpeg...")
            
            # Fallback to ffmpeg subprocess
            try:
                import subprocess
                import shutil
                
                # Check if ffmpeg is available
                ffmpeg_path = shutil.which('ffmpeg')
                if not ffmpeg_path:
                    raise FileNotFoundError(
                        "FFmpeg is required to process non-WAV audio files.\n"
                        "The browser recorded in a format that requires FFmpeg to decode.\n\n"
                        "Solutions:\n"
                        "  1. Install FFmpeg: https://ffmpeg.org/download.html\n"
                        "     - Download, extract, and add bin folder to PATH\n"
                        "     - Or use: winget install ffmpeg (Windows 10+)\n"
                        "  2. Use the updated frontend that records in WAV format\n"
                        "  3. Upload .wav files directly instead of recording"
                    )
                
                # Convert to WAV using ffmpeg
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp_path = tmp.name
                
                try:
                    subprocess.run([
                        ffmpeg_path, '-i', str(filepath),
                        '-ar', '16000', '-ac', '1', '-y',
                        tmp_path
                    ], capture_output=True, check=True)
                    
                    # Now read the converted WAV
                    return self._load_audio_file(tmp_path)
                finally:
                    if Path(tmp_path).exists():
                        Path(tmp_path).unlink()
                        
            except FileNotFoundError as e:
                raise ValueError(str(e))
            except Exception as e:
                raise ValueError(
                    f"Could not load audio file: {filepath}\n"
                    f"Supported: .wav (directly), other formats require ffmpeg or pydub\n"
                    f"Install pydub: pip install pydub\n"
                    f"Error: {e}"
                )
    
    def transcribe_stream(
        self,
        audio_stream: Generator,
        callback: Optional[callable] = None
    ) -> Generator[str, None, TranscriptionResult]:
        """
        Stream transcription for real-time processing
        
        Args:
            audio_stream: Generator yielding AudioData chunks
            callback: Optional callback for each partial result
            
        Yields:
            Partial transcription strings
            
        Returns:
            Final TranscriptionResult
        """
        if self.model is None:
            raise RuntimeError("No model loaded.")
        
        recognizer = self.vosk.KaldiRecognizer(self.model, self.sample_rate)
        recognizer.SetWords(True)
        recognizer.SetPartialWords(True)
        
        all_words = []
        utterances = []
        total_duration = 0.0
        
        for audio_chunk in audio_stream:
            samples = audio_chunk.samples
            total_duration += audio_chunk.duration
            
            if audio_chunk.sample_rate != self.sample_rate:
                samples = self._resample(samples, audio_chunk.sample_rate, self.sample_rate)
            
            audio_bytes = self._to_pcm_bytes(samples)
            
            if recognizer.AcceptWaveform(audio_bytes):
                result = json.loads(recognizer.Result())
                words, utterance = self._parse_result(result)
                all_words.extend(words)
                if utterance:
                    utterances.append(utterance)
                    if callback:
                        callback(utterance.text)
                    yield utterance.text
            else:
                # Partial result
                partial = json.loads(recognizer.PartialResult())
                if partial.get('partial'):
                    yield f"... {partial['partial']}"
        
        # Final result
        final_result = json.loads(recognizer.FinalResult())
        words, utterance = self._parse_result(final_result)
        all_words.extend(words)
        if utterance:
            utterances.append(utterance)
        
        full_text = ' '.join(w.text for w in all_words)
        
        return TranscriptionResult(
            text=full_text,
            utterances=utterances,
            words=all_words,
            duration=total_duration,
            language="en",
            model_name=self.model_name
        )
    
    def transcribe_file(
        self,
        filepath: str,
        show_progress: bool = True
    ) -> TranscriptionResult:
        """
        Transcribe directly from audio file
        
        Args:
            filepath: Path to audio file
            show_progress: Show progress
            
        Returns:
            TranscriptionResult
        """
        # Import here to avoid circular dependency
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from audio import AudioCapture
        
        capture = AudioCapture(target_sample_rate=self.sample_rate)
        audio_data = capture.load_file(filepath)
        
        return self.transcribe(audio_data, show_progress)
    
    def _parse_result(self, result: Dict) -> tuple:
        """Parse Vosk result into Word and Utterance objects"""
        words = []
        
        if 'result' in result:
            for w in result['result']:
                word = Word(
                    text=w.get('word', ''),
                    start_time=w.get('start', 0.0),
                    end_time=w.get('end', 0.0),
                    confidence=w.get('conf', 1.0)
                )
                words.append(word)
        
        utterance = None
        if words:
            text = result.get('text', ' '.join(w.text for w in words))
            utterance = Utterance(
                text=text.strip(),
                words=words,
                start_time=words[0].start_time,
                end_time=words[-1].end_time
            )
        
        return words, utterance
    
    def _to_pcm_bytes(self, samples: np.ndarray) -> bytes:
        """Convert float samples to 16-bit PCM bytes"""
        # Ensure float32
        samples = samples.astype(np.float32)
        
        # Clip to [-1, 1]
        samples = np.clip(samples, -1.0, 1.0)
        
        # Convert to 16-bit integer
        int_samples = (samples * 32767).astype(np.int16)
        
        return int_samples.tobytes()
    
    def _resample(self, samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        try:
            from scipy import signal
            num_samples = int(len(samples) * target_sr / orig_sr)
            return signal.resample(samples, num_samples)
        except ImportError:
            # Simple linear interpolation fallback
            ratio = target_sr / orig_sr
            indices = np.arange(0, len(samples), 1/ratio)
            indices = indices[indices < len(samples) - 1].astype(int)
            return samples[indices]
    
    @staticmethod
    def download_model(model_name: str, output_dir: str = "./models") -> str:
        """
        Download a Vosk model
        
        Args:
            model_name: One of 'en-us', 'en-us-small', 'en-in', 'hi', 'te'
            output_dir: Directory to save the model
            
        Returns:
            Path to downloaded model
        """
        import urllib.request
        import zipfile
        
        if model_name not in Transcriber.MODEL_URLS:
            available = list(Transcriber.MODEL_URLS.keys())
            raise ValueError(f"Unknown model: {model_name}. Available: {available}")
        
        url = Transcriber.MODEL_URLS[model_name]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        zip_path = output_dir / f"{model_name}.zip"
        
        print(f"[Transcriber] Downloading {model_name} from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        
        print(f"[Transcriber] Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        # Clean up zip
        zip_path.unlink()
        
        # Find extracted directory
        for item in output_dir.iterdir():
            if item.is_dir() and model_name.replace('-', '') in item.name.replace('-', ''):
                print(f"[Transcriber] Model ready at: {item}")
                return str(item)
        
        return str(output_dir)
