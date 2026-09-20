import io
from pathlib import Path
import struct
import wave

import pytest

from app.providers.demo import tone_wav
from app.services.audio import AudioEngine, AudioError, evaluate_qc, qc_text


def test_text_qc_detects_omission_repetition_and_preserves_russian_letters():
    assert qc_text("А́нна, всё хорошо!", "Анна все хорошо")["wer"] == 0
    assert qc_text("Мой дом", "Мои дом")["wer"] == .5
    missing = qc_text("Анна открыла дверь и вошла в дом", "Анна вошла в дом")
    assert missing["word_errors"] == 3
    repeated = qc_text("Анна пришла", "Анна пришла пришла пришла")
    assert repeated["wer"] == 1
    assert qc_text("Слова", "")["wer"] == 1


def test_qc_missing_evidence_never_passes():
    acoustic = {"duration_seconds": 1, "silence_ratio": .1, "clipping_ratio": 0, "peak": .5}
    text = qc_text("Всё хорошо", "Все хорошо")
    verified_speaker = {"verified": True, "method": "reference_embedding", "match": True}
    assert evaluate_qc(acoustic)["status"] == "REVIEW"
    assert evaluate_qc(acoustic, text)["status"] == "REVIEW"
    assert evaluate_qc(acoustic, text, {"speaker_id": "speaker_1"})["status"] == "REVIEW"
    assert evaluate_qc(acoustic, text, verified_speaker)["status"] == "PASS"
    assert evaluate_qc(acoustic, text, verified_speaker, demo=True)["status"] == "REVIEW"
    assert evaluate_qc(acoustic, qc_text("Один два три", "Один"), verified_speaker)["status"] == "FAIL"
    assert evaluate_qc({**acoustic, "silence_ratio": 1.0}, text, verified_speaker)["status"] == "FAIL"
    assert evaluate_qc({"duration_seconds": 1}, text, verified_speaker)["status"] == "REVIEW"


def test_real_acoustic_inspection(tmp_path):
    engine = AudioEngine()
    tone = tmp_path / "tone.wav"
    tone.write_bytes(tone_wav(duration=1))
    metrics = engine.inspect(tone)
    assert metrics["duration_seconds"] == pytest.approx(1, abs=.001)
    assert .1 < metrics["peak"] < .3
    assert 0 < metrics["silence_ratio"] < .7
    assert not metrics["clipping"]
    assert "speaker" not in metrics

    for name, value in [("silence", 0), ("clipped", 32767)]:
        path = tmp_path / (name + ".wav")
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(struct.pack("<h", value) * 24000)
        found = engine.inspect(path)
        assert found["all_silence"] if name == "silence" else found["clipping"]


def test_export_wav_mp3_m4b_audio_and_chapter_metadata(tmp_path):
    engine = AudioEngine()
    source = tmp_path / "тестовая запись.wav"
    source.write_bytes(tone_wav(duration=.7))
    chapters = [
        {"id": "chapter-one", "title": "Глава 1: Дом #1", "segments": [{"path": source, "pause_before_ms": 100, "pause_after_ms": 200}, {"path": source, "pause_before_ms": 0, "pause_after_ms": 100}]},
        {"id": "chapter-two", "title": "Глава 2 = Встреча;", "segments": [{"path": source, "pause_before_ms": 100, "pause_after_ms": 200}]},
    ]
    result = engine.export(chapters, tmp_path / "экспорт с пробелами", "Книга = тест", "Автор #1", ["wav", "mp3", "m4b"])
    assert [f["format"] for f in result["files"]].count("wav") == 2
    assert [f["format"] for f in result["files"]].count("mp3") == 2
    assert [f["format"] for f in result["files"]].count("m4b") == 1
    durations = {"chapter-one": 1.8, "chapter-two": 1.0}
    for item in result["files"]:
        path = Path(item["path"])
        assert path.stat().st_size > 100
        inspection = engine.inspect(path)
        assert inspection["duration_seconds"] == pytest.approx(durations.get(item.get("chapter_id"), 2.8), abs=.08)
        assert not inspection["all_silence"]
        if item["format"] == "wav":
            assert inspection["peak_dbfs"] <= -1.9
        if item["format"] == "m4b":
            metadata = engine._run(["-v", "error", "-i", str(path), "-f", "ffmetadata", "-"]).stdout.decode("utf-8")
            assert metadata.count("[CHAPTER]") == 2
            assert "Дом" in metadata and "Встреча" in metadata
            assert "Автор" in metadata and "Книга" in metadata
    assert all(item["method"].startswith("constant chapter gain") for item in result["mastering"])


def test_audio_errors_are_explicit(tmp_path):
    engine = AudioEngine(ffmpeg_path=str(tmp_path / "missing-ffmpeg"))
    with pytest.raises(AudioError, match="FFmpeg"):
        engine._run(["-version"])
    engine = AudioEngine()
    with pytest.raises(AudioError, match="Нет глав"):
        engine.export([], tmp_path, "Title", "Author", ["wav"])
    with pytest.raises(AudioError, match="Пауза"):
        AudioEngine._pause(None, -1)
    with pytest.raises(AudioError, match="нет аудио"):
        engine.export([{"id": "empty", "title": "Empty", "segments": []}], tmp_path, "Title", "Author", ["wav"])
