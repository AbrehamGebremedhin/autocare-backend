import re

import pytest

from app.services.kb import build_tsquery, chunk_pages, ingest_pdf, search


def test_build_tsquery_ors_keywords_from_chatty_message():
    q = build_tsquery("My car is overheating and the fan keeps clicking on and off")
    assert q == "car | overheating | fan | keeps | clicking | off"


def test_build_tsquery_empty_for_stopwords_only():
    assert build_tsquery("is it on?") == ""


def test_build_tsquery_only_emits_safe_characters():
    q = build_tsquery("fan'; drop table x; -- & | ! ( )")
    assert re.fullmatch(r"[a-z0-9 |]*", q)
    assert "fan" in q


def test_build_tsquery_dedups_and_caps_terms():
    assert build_tsquery("fan fan FAN belt") == "fan | belt"
    words = " ".join(f"word{i}" for i in range(30))
    assert len(build_tsquery(words, max_terms=5).split(" | ")) == 5


def test_chunk_pages_keeps_page_numbers_and_size():
    pages = ["a" * 1000 + "\n\n" + "b" * 1000, "", "c" * 4000]
    chunks = chunk_pages(pages, size=1500)
    assert [p for p, _ in chunks] == [1, 1, 3, 3, 3]
    assert all(len(t) <= 1500 for _, t in chunks)


def test_chunk_pages_packs_small_paragraphs_and_skips_blank():
    chunks = chunk_pages(["one\n\ntwo\n\n   \n\nthree", "   \n\n"], size=1500)
    assert len(chunks) == 1
    assert chunks[0][0] == 1
    assert all(w in chunks[0][1] for w in ("one", "two", "three"))


async def test_search_skips_rpc_for_empty_query(mocker):
    rpc = mocker.patch("app.services.kb._rpc")
    assert await search("is it on?", "toyota-echo-2001", 8) == []
    rpc.assert_not_called()


async def test_search_passes_car_and_k(mocker):
    rpc = mocker.patch("app.services.kb._rpc", return_value=[{"source_type": "manual", "title": "Echo", "page": 3, "content": "x", "rank": 0.5}])
    out = await search("radiator fan", "toyota-echo-2001", 8)
    rpc.assert_called_once_with("search_kb", {"tsq": "radiator | fan", "p_car_id": "toyota-echo-2001", "k": 8})
    assert out[0]["page"] == 3


async def test_search_maps_rows_to_passages(mocker):
    mocker.patch("app.services.kb._rpc", return_value=[{"source_type": "book", "car_id": None, "title": "T", "page": 1, "content": "c", "rank": 1.0}])
    out = await search("radiator fan", "x", 3)
    assert out == [{"source_type": "book", "title": "T", "page": 1, "content": "c"}]


def _fake_reader(mocker, pages):
    reader = mocker.patch("app.services.kb.PdfReader")
    reader.return_value.pages = [mocker.Mock(extract_text=mocker.Mock(return_value=t)) for t in pages]
    return reader


async def test_ingest_pdf_deletes_then_inserts_in_batches(mocker, tmp_path):
    calls = []
    mocker.patch("app.services.kb._delete_existing", side_effect=lambda t, c: calls.append(("delete", t, c)))
    mocker.patch("app.services.kb._insert_batch", side_effect=lambda rows: calls.append(("insert", len(rows))))
    # 1100 pages of one short chunk each
    _fake_reader(mocker, [f"page text {i}" for i in range(1100)])
    n = await ingest_pdf(tmp_path / "m.pdf", source_type="manual", title="Echo", car_id="toyota-echo-2001")
    assert n == 1100
    assert calls == [("delete", "Echo", "toyota-echo-2001"), ("insert", 500), ("insert", 500), ("insert", 100)]


async def test_ingest_pdf_row_shape(mocker, tmp_path):
    inserted = []
    mocker.patch("app.services.kb._delete_existing")
    mocker.patch("app.services.kb._insert_batch", side_effect=inserted.append)
    _fake_reader(mocker, ["hello"])
    await ingest_pdf(tmp_path / "b.pdf", source_type="book", title="Book")
    assert inserted == [[{"source_type": "book", "car_id": None, "title": "Book", "page": 1, "content": "hello"}]]


async def test_ingest_pdf_returns_zero_for_textless_pdf(mocker, tmp_path):
    insert = mocker.patch("app.services.kb._insert_batch")
    delete = mocker.patch("app.services.kb._delete_existing")
    _fake_reader(mocker, ["", None, "  \n "])
    assert await ingest_pdf(tmp_path / "s.pdf", source_type="book", title="Scan") == 0
    insert.assert_not_called()
    delete.assert_not_called()


async def test_ingest_pdf_strips_nul_characters(mocker, tmp_path):
    inserted = []
    mocker.patch("app.services.kb._delete_existing")
    mocker.patch("app.services.kb._insert_batch", side_effect=inserted.append)
    _fake_reader(mocker, ["a\x00b"])
    await ingest_pdf(tmp_path / "n.pdf", source_type="book", title="Nul")
    assert inserted[0][0]["content"] == "ab"


async def test_ingest_pdf_unreadable_pdf_leaves_existing_rows(mocker, tmp_path):
    delete = mocker.patch("app.services.kb._delete_existing")
    mocker.patch("app.services.kb.PdfReader", side_effect=ValueError("corrupt"))
    with pytest.raises(ValueError):
        await ingest_pdf(tmp_path / "bad.pdf", source_type="book", title="Bad")
    delete.assert_not_called()
