import io
import zipfile

import pytest

from app.parsers.books import MAX_UTTERANCE_CHARS, paragraph, parse_book, split_text


def paragraphs(book):
    return [item for chapter in book["chapters"] for scene in chapter["scenes"] for item in scene["paragraphs"]]


def make_epub(chapter_files=None, spine="second first", extra=None):
    chapter_files = chapter_files or {
        "first": "<h1>Глава первая</h1><p>Первый <em>абзац</em>.</p><p>Ещё один.</p>",
        "second": "<h1>Глава вторая</h1><div>Прямой текст</div><p>Второй абзац.</p>",
    }
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", '<container><rootfiles><rootfile full-path="OEBPS/book.opf"/></rootfiles></container>')
        items = ''.join(f'<item id="{name}" href="{name}.xhtml" media-type="application/xhtml+xml"/>' for name in chapter_files)
        order = ''.join(f'<itemref idref="{name}"/>' for name in spine.split())
        archive.writestr("OEBPS/book.opf", '<package xmlns:dc="http://purl.org/dc/elements/1.1/"><metadata><dc:title>Порядок</dc:title><dc:creator>Автор</dc:creator><dc:language>ru</dc:language></metadata><manifest>' + items + '</manifest><spine>' + order + '</spine></package>')
        for name, text in chapter_files.items():
            archive.writestr(f"OEBPS/{name}.xhtml", '<!DOCTYPE html><html xmlns="http://www.w3.org/1999/xhtml"><head><title>Служебный заголовок</title></head><body>' + text + '</body></html>')
        for name, data in (extra or {}).items():
            archive.writestr(name, data)
    return stream.getvalue()


@pytest.mark.parametrize("encoding", ["utf-8-sig", "cp1251", "utf-16"])
def test_txt_encoding_chapters_and_exact_source(encoding):
    lines = ["Предисловие.", "Глава 1", "  Вечер.  ", "— Привет, — сказала Анна, — я вернулась.", "Глава 2", "Конец."]
    result = parse_book(("\r\n\r\n".join(lines)).encode(encoding), "novel.txt")
    assert [item["text"] for item in paragraphs(result)] == lines
    assert [chapter["title"] for chapter in result["chapters"]] == ["Начало", "Глава 1", "Глава 2"]
    for item in paragraphs(result):
        assert ''.join(utterance["source_text"] for utterance in item["utterances"]) == item["text"]
    speech = paragraphs(result)[3]["utterances"]
    assert [item["type"] for item in speech] == ["UNKNOWN", "NARRATOR", "UNKNOWN"]
    assert all(item["speaker_confidence"] == 0 for item in speech if item["type"] == "UNKNOWN")


def test_quotes_are_uncertain_and_author_stays_narrator():
    text = 'Он сказал: «Не сегодня». А затем вышел.'
    result = paragraph(text)
    assert [item["type"] for item in result["utterances"]] == ["NARRATOR", "UNKNOWN", "NARRATOR"]
    assert ''.join(item["source_text"] for item in result["utterances"]) == text


def test_long_paragraph_preserved_and_segments_bounded():
    original = '  — ' + ('Необыкновенно длинная, но осмысленная фраза.  ' * 130) + ('x' * 1900)
    result = paragraph(original)
    assert result["text"] == original
    assert ''.join(item["source_text"] for item in result["utterances"]) == original
    assert max(len(item["source_text"]) for item in result["utterances"]) <= MAX_UTTERANCE_CHARS
    assert all(item["type"] == "UNKNOWN" for item in result["utterances"])
    assert ''.join(split_text(original, 37)) == original


def test_fb2_nested_chapters_notes_poetry_and_inline_markup():
    xml = '''<?xml version="1.0" encoding="UTF-8"?>
    <FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
      <description><title-info><author><first-name>Иван</first-name><last-name>Автор</last-name></author><book-title>Книга</book-title><lang>ru</lang></title-info></description>
      <body><section><title><p>Часть I</p></title><p>Начало <emphasis>истории</emphasis>.</p>
      <section><title><p>Глава 1</p></title><p>— Кто здесь?</p><poem><stanza><v>Первая строка</v><v>Вторая строка</v></stanza></poem></section><p>Послесловие части.</p></section></body>
      <body name="notes"><section><p>Примечание.</p></section></body>
    </FictionBook>'''
    book = parse_book(xml.encode(), "test.fb2")
    assert book["title"] == "Книга"
    assert book["author"] == "Иван Автор"
    assert [item["text"] for item in paragraphs(book)] == ["Часть I", "Начало истории.", "Глава 1", "— Кто здесь?", "Первая строка", "Вторая строка", "Послесловие части.", "Примечание."]


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
def test_xml_entities_rejected_even_utf16(encoding):
    xml = f'<?xml version="1.0" encoding="{encoding}"?><!DOCTYPE FictionBook [<!ENTITY secret SYSTEM "file:///etc/passwd">]><FictionBook><body><section><p>&secret;</p></section></body></FictionBook>'
    with pytest.raises(ValueError, match="XML"):
        parse_book(xml.encode(encoding), "unsafe.fb2")


def test_epub_spine_order_metadata_and_no_duplicate_paragraphs():
    book = parse_book(make_epub(), "book.epub")
    assert book["title"] == "Порядок"
    assert book["author"] == "Автор"
    assert [chapter["title"] for chapter in book["chapters"]] == ["Глава вторая", "Глава первая"]
    assert [item["text"] for item in paragraphs(book)] == ["Глава вторая", "Прямой текст", "Второй абзац.", "Глава первая", "Первый абзац.", "Ещё один."]


def test_epub_preserves_container_inline_text_and_line_breaks():
    book = parse_book(make_epub({"first": "<div>Текст <em>курсивом</em> продолжен.<br/>Новая строка.</div><blockquote><p>Цитата.</p></blockquote>"}, "first"), "book.epub")
    assert [item["text"] for item in paragraphs(book)] == ["Текст курсивом продолжен.\nНовая строка.", "Цитата."]


@pytest.mark.parametrize("path", ["../outside.txt", "/absolute.txt", "C:/drive.txt", "bad\\path.txt"])
def test_epub_unsafe_archive_paths_rejected(path):
    content = make_epub(extra={path: "malicious"})
    # On Windows ZipInfo normalizes backslashes while writing; model a foreign archive.
    if "\\" in path:
        content = content.replace(path.replace("\\", "/").encode(), path.encode())
    with pytest.raises(ValueError, match="путь"):
        parse_book(content, "bad.epub")


def test_epub_zip_bomb_and_missing_spine_reference_rejected():
    with pytest.raises(ValueError, match="сжатия"):
        parse_book(make_epub(extra={"bomb.txt": "x" * (2 * 1024 * 1024)}), "bad.epub")
    with pytest.raises(ValueError, match="itemref"):
        parse_book(make_epub(spine="missing"), "bad.epub")


@pytest.mark.parametrize("content,name", [(b"", "empty.txt"), (b"  \n  ", "blank.txt"), (b"abc\x00def", "binary.txt"), (b"abc", "bad.docx"), (b"bad", "bad.epub")])
def test_reject_empty_binary_and_unsupported(content, name):
    with pytest.raises(ValueError):
        parse_book(content, name)
