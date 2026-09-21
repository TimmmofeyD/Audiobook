# Changelog

## 2026-09-21

### Pipeline
- One-click «Аудио-пайплайн»: analyze (all chapters) → wiki research → direction → casting → generation
- Per-chapter analysis retry up to 3 attempts with backoff
- Honest failure reporting: pipeline job FAILS when analysis fails instead of fake success
- Sub-jobs bypass queue uniqueness check (no more 409 self-block)
- Guard against None character/scene records from repository

### Character analysis
- Name confirmation across three forms: full name, any token ≥4 chars, name without spaces
- Name normalization: case, ё/е, й/и, apostrophes, hyphens (fixes «Эль'Джонсон»/«Анастана» failures)
- LLM JSON repair: markdown fences, duplicate objects, unclosed brackets, truncation recovery
- Narrator default: male, middle-aged (40 y.o.) voice brief

### Wiki researcher
- Fandom wiki enrichment with canon summary, source URL
- Age band extraction (child/young_adult/adult/middle_aged/elderly)
- Emperor/Primarchs rule: middle_aged — «sounds like 30–40 y.o., timeless»

### Direction
- Unique per-utterance delivery enforced; template outputs trigger retry
- Director prompt hardened: no repeated delivery strings across utterances

### UI
- Job cancel button («Отменить задачу») for RUNNING/QUEUED jobs
- Delete book button with 409 guard while jobs active
- Per-chapter «Анализ» button, «Вики: персонаж» in casting
- Audio pipeline button on overview

### API
- DELETE /api/books/{id} — removes book data, preserves DB schema
- POST /api/jobs/{id}/cancel — user cancellation
- POST /api/books/{id}/pipeline — full pipeline orchestration
- POST /api/characters/{id}/research — wiki enrichment

### Infra
- LLM switched to Funpay Gateway (buymirror.duckdns.org)
- Keys excluded from repo (.env git-ignored, .env.example clean)
