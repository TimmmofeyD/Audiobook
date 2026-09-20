"""Real audio inspection and chapter mastering with FFmpeg.

QC is deliberately incomplete without independent ASR and speaker evidence.
Chapter gain is constant: quiet/shouted passages retain their relative levels.
"""
from __future__ import annotations

from array import array
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import wave


class AudioError(RuntimeError):
    pass


def _normal_text(text: str) -> str:
    text = unicodedata.normalize("NFD", text.casefold().replace("ё", "е"))
    # Ignore stress marks but preserve Russian й (и + combining breve in NFD).
    text = unicodedata.normalize("NFC", "".join(ch for ch in text if ch not in {"\u0301", "\u0300"}))
    return " ".join(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


def _distance(a: list | str, b: list | str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def qc_text(expected: str, actual: str) -> dict:
    reference, hypothesis = _normal_text(expected), _normal_text(actual)
    reference_words, hypothesis_words = reference.split(), hypothesis.split()
    word_errors = _distance(reference_words, hypothesis_words)
    char_errors = _distance(reference, hypothesis)
    return {"wer": word_errors / max(1, len(reference_words)), "cer": char_errors / max(1, len(reference)), "word_errors": word_errors, "character_errors": char_errors, "expected_words": len(reference_words), "actual_words": len(hypothesis_words), "expected_normalized": reference, "actual_normalized": hypothesis, "transcript": actual}


def evaluate_qc(acoustic: dict, text_metrics: dict | None = None, speaker_metrics: dict | None = None, demo: bool = False) -> dict:
    issues, failures = [], []
    duration = acoustic.get("duration_seconds", acoustic.get("duration", 0))
    if not acoustic or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        failures.append("Аудио пусто или не удалось измерить длительность.")
    if any(key not in acoustic for key in ("silence_ratio", "clipping_ratio", "peak")):
        issues.append("Акустические измерения неполны.")
    if acoustic.get("clipping") or acoustic.get("clipping_ratio", 0) > .001:
        failures.append("Обнаружен цифровой клиппинг.")
    if acoustic.get("all_silence") or acoustic.get("silence_ratio", 0) >= .98:
        failures.append("Почти весь фрагмент состоит из тишины.")
    elif acoustic.get("silence_ratio", 0) > .65:
        issues.append("Необычно высокая доля тишины; требуется прослушивание.")
    if text_metrics is None:
        issues.append("ASR не выполнен; точность произнесённого текста не проверена.")
    else:
        wer, cer = text_metrics.get("wer"), text_metrics.get("cer")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in (wer, cer)):
            issues.append("ASR-метрики неполны или некорректны.")
        elif wer > .25 or cer > .18:
            failures.append("ASR обнаружил существенные пропуски, повторы или искажения текста.")
        elif wer > .08 or cer > .05:
            issues.append("Текст ASR отличается от исходного; проверьте произношение.")
    # Diarization labels alone do not establish the identity of a saved voice.
    if not speaker_metrics or speaker_metrics.get("verified") is not True or not speaker_metrics.get("method"):
        issues.append("Идентичность голоса не проверена speaker-encoder; требуется прослушивание.")
    elif speaker_metrics.get("match") is not True:
        failures.append("Проверка идентичности голоса не подтвердила совпадение с эталоном.")
    if demo:
        issues.append("ДЕМО: тестовые тоны не содержат речи. Автоматический PASS невозможен.")
    return {"status": "FAIL" if failures else "REVIEW" if issues else "PASS", "issues": failures + issues, "metrics": {"acoustic": acoustic, "text": text_metrics, "speaker": speaker_metrics, "demo": demo}}


def _metadata(value: str) -> str:
    # FFmetadata reserves backslash, =, ;, # and newline.
    return str(value).replace("\\", "\\\\").replace("=", "\\=").replace(";", "\\;").replace("#", "\\#").replace("\r", " ").replace("\n", " ")


class AudioEngine:
    def __init__(self, ffmpeg_path: str = ""):
        self.ffmpeg_path = ffmpeg_path or shutil.which("ffmpeg") or ""
        if not self.ffmpeg_path:
            try:
                import imageio_ffmpeg
                self.ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            except (ImportError, RuntimeError):
                pass

    def _run(self, args: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
        if not self.ffmpeg_path:
            raise AudioError("FFmpeg не найден. Установите imageio-ffmpeg или задайте FFMPEG_PATH.")
        try:
            result = subprocess.run([self.ffmpeg_path, "-hide_banner", "-nostdin", "-y", *map(str, args)], capture_output=True, timeout=timeout, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise AudioError("Не удалось запустить или дождаться FFmpeg.") from error
        if result.returncode:
            details = result.stderr.decode("utf-8", errors="replace")[-1500:]
            raise AudioError("Ошибка обработки аудио FFmpeg: " + details)
        return result

    def _decode(self, source: Path, destination: Path, master: bool = False):
        source = Path(source)
        if not source.is_file():
            raise AudioError(f"Аудиофайл не найден: {source.name}")
        args = ["-v", "error", "-i", str(source), "-map", "0:a:0", "-vn"]
        if master:
            args += ["-ar", "44100", "-ac", "1"]
        self._run(args + ["-c:a", "pcm_s16le", str(destination)])

    def inspect(self, path: Path) -> dict:
        with tempfile.TemporaryDirectory(prefix="audiobook-qc-") as temporary:
            decoded = Path(temporary) / "decoded.wav"
            self._decode(Path(path), decoded)
            with wave.open(str(decoded), "rb") as wav:
                rate, channels, frames = wav.getframerate(), wav.getnchannels(), wav.getnframes()
                peak, clipped, square_sum, count, silent_frames, total_frames = 0, 0, 0, 0, 0, 0
                window = max(1, rate // 50)  # 20 ms RMS windows, not zero crossings.
                while data := wav.readframes(window):
                    samples = array("h", data)
                    if sys.byteorder != "little":
                        samples.byteswap()
                    if not samples:
                        continue
                    squares = sum(value * value for value in samples)
                    square_sum += squares
                    count += len(samples)
                    peak = max(peak, max(abs(value) for value in samples))
                    clipped += sum(abs(value) >= 32760 for value in samples)
                    window_frames = len(samples) // channels
                    total_frames += window_frames
                    if math.sqrt(squares / len(samples)) / 32768 < 10 ** (-45 / 20):
                        silent_frames += window_frames
        rms = math.sqrt(square_sum / max(1, count)) / 32768
        peak_fraction = peak / 32768
        silence_ratio = silent_frames / max(1, total_frames)
        return {"duration_seconds": frames / rate, "duration": frames / rate, "sample_rate": rate, "channels": channels, "peak": peak_fraction, "peak_dbfs": 20 * math.log10(peak_fraction) if peak_fraction else -120.0, "rms_dbfs": 20 * math.log10(rms) if rms else -120.0, "silence_ratio": silence_ratio, "clipping_ratio": clipped / max(1, count), "clipping": clipped / max(1, count) > .001, "all_silence": silence_ratio >= .98, "silence_threshold_dbfs": -45, "analysis": "PCM acoustic checks; no speaker identity or perceptual artifact detection"}

    @staticmethod
    def _pause(wav, milliseconds):
        if isinstance(milliseconds, bool) or not isinstance(milliseconds, (int, float)) or not math.isfinite(milliseconds) or not 0 <= milliseconds <= 30000:
            raise AudioError("Пауза должна быть числом от 0 до 30000 мс.")
        remaining = round(44100 * milliseconds / 1000)
        while remaining:
            count = min(remaining, 44100)
            wav.writeframesraw(b"\x00\x00" * count)
            remaining -= count

    def _assemble_chapter(self, chapter: dict, path: Path, temporary: Path):
        if not chapter.get("segments"):
            raise AudioError(f"В главе «{chapter.get('title', '')}» нет аудиофрагментов.")
        with wave.open(str(path), "wb") as destination:
            destination.setnchannels(1)
            destination.setsampwidth(2)
            destination.setframerate(44100)
            for segment in chapter["segments"]:
                self._pause(destination, segment.get("pause_before_ms", 0))
                decoded = temporary / "segment.wav"
                self._decode(Path(segment["path"]), decoded, master=True)
                with wave.open(str(decoded), "rb") as source:
                    while chunk := source.readframes(65536):
                        destination.writeframesraw(chunk)
                self._pause(destination, segment.get("pause_after_ms", 0))

    def _master_gain(self, source: Path) -> dict:
        result = self._run(["-i", str(source), "-af", "loudnorm=I=-20:TP=-2:LRA=11:print_format=json", "-f", "null", "-"])
        output = result.stderr.decode("utf-8", errors="replace")
        match = re.search(r'\{\s*"input_i".*?\}', output, flags=re.S)
        if not match:
            raise AudioError("FFmpeg не вернул измерение громкости главы.")
        measurement = json.loads(match.group())
        integrated, true_peak = float(measurement["input_i"]), float(measurement["input_tp"])
        gain = min(-20.0 - integrated, -2.0 - true_peak) if math.isfinite(integrated) and math.isfinite(true_peak) else 0.0
        return {"gain_db": gain, "measured_lufs": integrated if math.isfinite(integrated) else None, "measured_true_peak_db": true_peak if math.isfinite(true_peak) else None, "target_lufs": -20, "peak_ceiling_db": -2, "method": "constant chapter gain, constrained by true peak; preserves dynamics"}

    def export(self, chapters: list[dict], output_dir: Path, title: str, author: str, formats: list[str]) -> dict:
        if not chapters:
            raise AudioError("Нет глав для экспорта.")
        requested = set(formats)
        if not requested or requested - {"wav", "mp3", "m4b"}:
            raise AudioError("Поддерживаются форматы wav, mp3, m4b.")
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        files, chapter_paths, chapter_frames, mastering = [], [], [], []
        with tempfile.TemporaryDirectory(prefix="mastering-", dir=output_dir) as temporary_name:
            temporary = Path(temporary_name)
            for number, chapter in enumerate(chapters, 1):
                raw = temporary / "chapter.wav"
                self._assemble_chapter(chapter, raw, temporary)
                gain = self._master_gain(raw)
                mastering.append({"chapter_id": chapter["id"], **gain})
                name = f"chapter_{number:04d}"
                master = output_dir / (name + ".wav")
                metadata = ["-metadata", f"title={chapter['title']}", "-metadata", f"album={title}", "-metadata", f"artist={author}", "-metadata", f"track={number}/{len(chapters)}"]
                self._run(["-v", "error", "-i", str(raw), "-af", f"volume={gain['gain_db']:.8f}dB", "-c:a", "pcm_s16le", *metadata, str(master)])
                chapter_paths.append(master)
                with wave.open(str(master), "rb") as wav:
                    chapter_frames.append(wav.getnframes())
                # WAV masters are always retained alongside requested deliveries.
                files.append({"path": str(master), "filename": master.name, "format": "wav", "chapter_id": chapter["id"]})
                if "mp3" in requested:
                    mp3 = output_dir / (name + ".mp3")
                    self._run(["-v", "error", "-i", str(master), "-c:a", "libmp3lame", "-b:a", "128k", "-id3v2_version", "3", *metadata, str(mp3)])
                    files.append({"path": str(mp3), "filename": mp3.name, "format": "mp3", "chapter_id": chapter["id"]})
            if "m4b" in requested:
                concat = temporary / "chapters.ffconcat"
                concat.write_text("ffconcat version 1.0\n" + "".join("file '" + str(path.as_posix()).replace("'", "'\\''") + "'\n" for path in chapter_paths), encoding="utf-8")
                metadata_path = temporary / "book.ffmetadata"
                metadata_text = f";FFMETADATA1\ntitle={_metadata(title)}\nartist={_metadata(author)}\nalbum={_metadata(title)}\ngenre=Audiobook\n"
                start = 0
                for chapter, frames in zip(chapters, chapter_frames):
                    end = start + frames
                    metadata_text += f"[CHAPTER]\nTIMEBASE=1/44100\nSTART={start}\nEND={end}\ntitle={_metadata(chapter['title'])}\n"
                    start = end
                metadata_path.write_text(metadata_text, encoding="utf-8")
                m4b = output_dir / "audiobook.m4b"
                self._run(["-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat), "-f", "ffmetadata", "-i", str(metadata_path), "-map", "0:a:0", "-map_metadata", "1", "-map_chapters", "1", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", "-f", "mp4", str(m4b)], timeout=14400)
                files.append({"path": str(m4b), "filename": m4b.name, "format": "m4b"})
        return {"files": files, "mastering": mastering}
