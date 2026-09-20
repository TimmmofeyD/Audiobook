"""Import literary text without discarding source paragraphs.

The original file remains the byte-level source of truth. Paragraph text preserves
decoded text (XML entity decoding and XML newline normalization are unavoidable).
Utterance source slices always concatenate exactly to their parent paragraph.
"""

from __future__ import annotations

import io
import posixpath
import re
import zipfile
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit

from defusedxml import ElementTree as SafeET
from defusedxml.common import DefusedXmlException
from xml.etree.ElementTree import ParseError

MAX_INPUT_BYTES = 25 * 1024 * 1024
MAX_ARCHIVE_BYTES = 80 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 3000
MAX_XML_NODES = 250_000
MAX_UTTERANCE_CHARS = 1700
MAX_SCENE_CHARS = 12_000
MAX_SCENE_UTTERANCES = 50

_CHAPTER = re.compile(r"^\s*(?:глава|часть|книга|chapter|part|book)\s+[\wIVXLCDMА-Яа-яЁё]+(?:[.\s:].*)?$|^\s*(?:пролог|эпилог|prologue|epilogue)\s*$", re.IGNORECASE)
_SCENE = re.compile(r"^\s*(?:\*\s*){3,}$|^\s*#\s*#\s*#\s*$")
_DIALOGUE_START = re.compile(r"^\s*[—–-]\s+")
_AUTHOR_REMARK = re.compile(r"\s+[—–]\s+(?:(?:тихо|громко|сухо|медленно|быстро|спокойно|вдруг|тихонько)\s+)?(?:сказал[аи]?|ответил[аи]?|спросил[аи]?|прошептал[аи]?|воскликнул[аи]?|проговорил[аи]?|добавил[аи]?|заметил[аи]?|крикнул[аи]?|произн[её]с(?:ла|ли)?|подумал[аи]?)\b", re.IGNORECASE)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _xml(content: bytes):
    try:
        root = SafeET.fromstring(content, forbid_entities=True, forbid_external=True)
    except (ParseError, DefusedXmlException, ValueError, LookupError) as exc:
        raise ValueError("Некорректный или небезопасный XML в книге") from exc
    stack = [(root, 0)]
    count = 0
    while stack:
        node, depth = stack.pop()
        count += 1
        if depth > 100 or count > MAX_XML_NODES:
            raise ValueError("Слишком сложная XML-структура книги")
        stack.extend((child, depth + 1) for child in node)
    return root


def _text(node) -> str:
    """Inline markup has no synthesized whitespace; explicit BR has a newline."""
    parts = [node.text or ""]
    for child in node:
        if _local(child.tag) == "br":
            parts.append("\n")
        elif _local(child.tag) not in {"script", "style", "binary"}:
            parts.append(_text(child))
        parts.append(child.tail or "")
    return "".join(parts)


def split_text(text: str, limit: int = MAX_UTTERANCE_CHARS) -> list[str]:
    """Bound a TTS segment while preserving every input character."""
    if limit < 1:
        raise ValueError("limit must be positive")
    result = []
    while len(text) > limit:
        window = text[:limit]
        lower_bound = max(1, limit // 3)
        ends = [m.end() for m in re.finditer(r"[.!?…][\"»”')]*\s+", window) if m.end() >= lower_bound]
        if ends:
            cut = ends[-1]
        else:
            spaces = [m.end() for m in re.finditer(r"\s+", window) if m.end() >= lower_bound]
            cut = spaces[-1] if spaces else limit
        result.append(text[:cut])
        text = text[cut:]
    if text:
        result.append(text)
    return result


def _utterance(source: str, kind: str) -> dict:
    spoken = source.strip()
    if kind == "UNKNOWN":
        spoken = re.sub(r"^[—–-]\s+", "", spoken)
    elif kind == "NARRATOR":
        # Author remarks retain punctuation in source; the boundary dash is not speech.
        spoken = re.sub(r"^[—–]\s+", "", spoken)
    return {"source_text": source, "spoken_text": spoken, "type": kind,
            "speaker_confidence": 1.0 if kind == "NARRATOR" else 0.0}


def paragraph(text: str, *, heading: bool = False) -> dict:
    pieces: list[tuple[str, str]] = []
    if not heading and _DIALOGUE_START.match(text):
        remark = _AUTHOR_REMARK.search(text)
        if remark:
            pieces.append((text[:remark.start()], "UNKNOWN"))
            # Only a sentence-ending author remark followed by another em dash
            # provides a safe boundary back into speech.
            rest = text[remark.start():]
            resumed = re.search(r"(?<=[.!?…,:;])\s+[—–]\s+", rest)
            if resumed:
                pieces.extend([(rest[:resumed.start()], "NARRATOR"), (rest[resumed.start():], "UNKNOWN")])
            else:
                pieces.append((rest, "NARRATOR"))
        else:
            pieces.append((text, "UNKNOWN"))
    elif not heading and "«" in text and "»" in text and (text.lstrip().startswith("«") or re.search(r":\s*«", text)):
        opening, closing = text.index("«"), text.rindex("»") + 1
        if opening:
            pieces.append((text[:opening], "NARRATOR"))
        pieces.append((text[opening:closing], "UNKNOWN"))
        if closing < len(text):
            pieces.append((text[closing:], "NARRATOR"))
    elif not heading and text.lstrip().startswith(("«", "“", '"')):
        # A quote may be dialogue, an inner thought, letter, or quotation. Review it.
        pieces.append((text, "UNKNOWN"))
    else:
        pieces.append((text, "NARRATOR"))
    utterances = [_utterance(chunk, kind) for source, kind in pieces for chunk in split_text(source)]
    assert "".join(item["source_text"] for item in utterances) == text
    return {"text": text, "utterances": utterances}


def _chapter(title: str, paragraphs: list[dict]) -> dict:
    scenes = []
    current: list[dict] = []
    chars = 0
    utterances = 0
    for item in paragraphs:
        count = len(item["utterances"])
        if current and (_SCENE.match(item["text"]) or chars + len(item["text"]) > MAX_SCENE_CHARS or utterances + count > MAX_SCENE_UTTERANCES):
            scenes.append({"paragraphs": current})
            current, chars, utterances = [], 0, 0
        current.append(item)
        chars += len(item["text"])
        utterances += count
    if current:
        scenes.append({"paragraphs": current})
    return {"title": title, "scenes": scenes}


def _txt(content: bytes, filename: str) -> dict:
    if content.startswith((b"\xff\xfe", b"\xfe\xff")):
        encoding = "utf-16"
    else:
        encoding = "utf-8-sig"
    try:
        text = content.decode(encoding)
    except UnicodeDecodeError:
        try:
            text = content.decode("cp1251")
        except UnicodeDecodeError as exc:
            raise ValueError("TXT должен быть в UTF-8, UTF-16 или Windows-1251") from exc
    if "\x00" in text or any(ord(char) < 32 and char not in "\r\n\t\f" for char in text):
        raise ValueError("TXT содержит бинарные или управляющие символы")
    chapters, current = [], []
    title = PurePosixPath(filename.replace("\\", "/")).stem
    chapter_title = "Начало"
    for line in text.splitlines():
        if not line.strip():
            continue
        heading = bool(_CHAPTER.match(line))
        if heading:
            if current:
                chapters.append(_chapter(chapter_title, current))
                current = []
            chapter_title = line.strip()
        item = paragraph(line, heading=heading)
        if _SCENE.match(line):
            item["utterances"] = []
        current.append(item)
    if current:
        chapters.append(_chapter(chapter_title, current))
    return {"title": title, "author": "", "language": "ru", "chapters": chapters}


def _fb2(content: bytes, filename: str) -> dict:
    root = _xml(content)
    if _local(root.tag) != "fictionbook":
        raise ValueError("Файл не содержит FictionBook")
    info = next((node for node in root.iter() if _local(node.tag) == "title-info"), None)
    title = PurePosixPath(filename).stem
    author = ""
    language = "ru"
    if info is not None:
        authors = []
        for node in info:
            tag = _local(node.tag)
            if tag == "book-title" and _text(node).strip():
                title = _text(node).strip()
            elif tag == "lang" and _text(node).strip():
                language = _text(node).strip()
            elif tag == "author":
                name = " ".join(_text(part).strip() for part in node if _local(part.tag) in {"first-name", "middle-name", "last-name"} and _text(part).strip())
                if not name:
                    name = " ".join(_text(part).strip() for part in node if _local(part.tag) == "nickname")
                if name:
                    authors.append(name)
        author = ", ".join(authors)
    chapters = []

    def parse_container(container, fallback):
        current = []
        chapter_title = fallback
        own_title = next((child for child in container if _local(child.tag) == "title"), None)
        if own_title is not None and _text(own_title).strip():
            chapter_title = " ".join(_text(own_title).split())
        for node in container:
            tag = _local(node.tag)
            if tag == "section":
                if current:
                    chapters.append(_chapter(chapter_title, current))
                    current = []
                parse_container(node, chapter_title)
            elif tag in {"image", "binary", "empty-line"}:
                continue
            else:
                for text, heading in _blocks(node, heading=tag in {"title", "subtitle"}):
                    if text.strip():
                        current.append(paragraph(text, heading=heading))
        if current:
            chapters.append(_chapter(chapter_title, current))

    for body in (node for node in root if _local(node.tag) == "body"):
        parse_container(body, body.attrib.get("name") or "Начало")
    return {"title": title, "author": author, "language": language, "chapters": chapters}


_BLOCK_TAGS = {"p", "v", "subtitle", "text-author", "date", "h1", "h2", "h3", "h4", "h5", "h6", "dt", "dd", "pre"}
_SKIP_TAGS = {"script", "style", "head", "binary", "image", "img", "svg"}


def _blocks(node, *, heading=False):
    """Read each leaf prose block once, including text in wrapper elements."""
    tag = _local(node.tag)
    if tag in _SKIP_TAGS:
        return
    heading = heading or tag in {"h1", "h2", "h3", "h4", "h5", "h6", "title", "subtitle"}
    if tag in _BLOCK_TAGS or not list(node):
        text = _text(node)
        if text.strip():
            yield text, heading
        return
    # A container made entirely of inline children is itself a paragraph.
    containers = _BLOCK_TAGS | {"div", "section", "article", "main", "blockquote", "li", "ul", "ol", "table", "tr", "td", "body", "title", "poem", "stanza", "epigraph", "cite"}
    if not any(_local(child.tag) in containers for child in node):
        text = _text(node)
        if text.strip():
            yield text, heading
        return
    if node.text and node.text.strip():
        yield node.text, heading
    for child in node:
        yield from _blocks(child, heading=heading)
        if child.tail and child.tail.strip():
            yield child.tail, heading


def _archive_path(path: str) -> str:
    if "\\" in path or "\x00" in path or path.startswith("/") or ":" in path:
        raise ValueError("Небезопасный путь в EPUB")
    if ".." in PurePosixPath(path).parts:
        raise ValueError("Небезопасный путь в EPUB")
    return posixpath.normpath(path)


def _epub(content: bytes, filename: str) -> dict:
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise ValueError("EPUB должен быть корректным ZIP-архивом") from exc
    with archive:
        entries = archive.infolist()
        if len(entries) > MAX_ARCHIVE_ENTRIES or sum(entry.file_size for entry in entries) > MAX_ARCHIVE_BYTES:
            raise ValueError("EPUB превышает безопасный размер распаковки")
        names = set()
        for entry in entries:
            # ZipInfo normalizes backslashes on Windows; validate its original spelling.
            _archive_path(entry.orig_filename)
            path = _archive_path(entry.filename)
            if path in names:
                raise ValueError("EPUB содержит дублирующиеся пути")
            names.add(path)
            if entry.flag_bits & 1:
                raise ValueError("Зашифрованные EPUB не поддерживаются")
            if entry.file_size > MAX_INPUT_BYTES or (entry.file_size > 1024 * 1024 and entry.file_size / max(entry.compress_size, 1) > 200):
                raise ValueError("Подозрительная степень сжатия EPUB")

        def read(path):
            path = _archive_path(path)
            try:
                with archive.open(path) as stream:
                    data = stream.read(MAX_INPUT_BYTES + 1)
            except (KeyError, RuntimeError, zipfile.BadZipFile, NotImplementedError) as exc:
                raise ValueError("EPUB ссылается на отсутствующий или повреждённый ресурс") from exc
            if len(data) > MAX_INPUT_BYTES:
                raise ValueError("Ресурс EPUB слишком велик")
            return data

        container = _xml(read("META-INF/container.xml"))
        rootfile = next((node.attrib.get("full-path") for node in container.iter() if _local(node.tag) == "rootfile"), None)
        if not rootfile:
            raise ValueError("В EPUB отсутствует package document")
        package_path = _archive_path(rootfile)
        package = _xml(read(package_path))
        title = PurePosixPath(filename).stem
        authors = []
        language = "ru"
        for metadata in (node for node in package if _local(node.tag) == "metadata"):
            for node in metadata:
                tag = _local(node.tag)
                if tag == "title" and _text(node).strip():
                    title = _text(node).strip()
                elif tag == "creator" and _text(node).strip():
                    authors.append(_text(node).strip())
                elif tag == "language" and _text(node).strip():
                    language = _text(node).strip()
        manifest = {node.attrib.get("id"): node.attrib for node in package.iter() if _local(node.tag) == "item"}
        chapters = []
        seen = set()
        for itemref in (node for node in package.iter() if _local(node.tag) == "itemref"):
            item = manifest.get(itemref.attrib.get("idref"))
            if not item:
                raise ValueError("EPUB spine содержит неизвестный itemref")
            if item.get("media-type") not in {"application/xhtml+xml", "text/html"}:
                continue
            href = urlsplit(item.get("href", ""))
            if href.scheme or href.netloc:
                raise ValueError("Внешние ресурсы в EPUB spine запрещены")
            relative = unquote(href.path)
            # Relative links may legitimately go up within the archive, but never escape it.
            path = posixpath.normpath(posixpath.join(posixpath.dirname(package_path), relative))
            path = _archive_path(path)
            if path in seen:
                continue
            seen.add(path)
            document = _xml(read(path))
            body = next((node for node in document.iter() if _local(node.tag) == "body"), document)
            current = []
            chapter_title = f"Глава {len(chapters) + 1}"
            for text, heading in _blocks(body):
                if heading:
                    if current:
                        chapters.append(_chapter(chapter_title, current))
                        current = []
                    chapter_title = " ".join(text.split())
                current.append(paragraph(text, heading=heading))
            if current:
                chapters.append(_chapter(chapter_title, current))
        return {"title": title, "author": ", ".join(authors), "language": language, "chapters": chapters}


def parse_book(content: bytes, filename: str) -> dict:
    if not content:
        raise ValueError("Файл книги пуст")
    if len(content) > MAX_INPUT_BYTES:
        raise ValueError("Размер книги превышает 25 МБ")
    extension = PurePosixPath(filename.lower()).suffix
    parser = {".txt": _txt, ".fb2": _fb2, ".epub": _epub}.get(extension)
    if parser is None:
        raise ValueError("Поддерживаются книги TXT, FB2 и EPUB")
    book = parser(content, filename)
    if not book["chapters"] or not any(item["utterances"] for chapter in book["chapters"] for scene in chapter["scenes"] for item in scene["paragraphs"]):
        raise ValueError("В книге не найден текст для озвучки")
    return book
