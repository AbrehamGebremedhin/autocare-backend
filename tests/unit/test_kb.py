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


async def test_search_skips_the_database_for_empty_query(fake_pool):
    assert await search("is it on?", "toyota-echo-2001", 8) == []
    assert fake_pool.calls == []


async def test_search_sends_exact_sql_and_params(fake_pool):
    fake_pool.rows = [{"source_type": "manual", "title": "Echo", "page": 3, "content": "x", "rank": 0.5}]
    out = await search("radiator fan", "toyota-echo-2001", 8)
    assert fake_pool.calls == [
        ("fetch", "SELECT * FROM search_kb($1, $2, $3)", ("radiator | fan", "toyota-echo-2001", 8))
    ]
    assert out[0]["page"] == 3


async def test_search_maps_rows_to_passages(fake_pool):
    fake_pool.rows = [{"source_type": "book", "car_id": None, "title": "T", "page": 1, "content": "c", "rank": 1.0}]
    out = await search("radiator fan", "x", 3)
    assert out == [{"source_type": "book", "title": "T", "page": 1, "content": "c"}]


def _fake_reader(mocker, pages):
    reader = mocker.patch("app.services.kb.PdfReader")
    reader.return_value.pages = [mocker.Mock(extract_text=mocker.Mock(return_value=t)) for t in pages]
    return reader


def _batches(fake_pool):
    return [c for c in fake_pool.calls if c[0] == "executemany"]


async def test_ingest_pdf_deletes_then_inserts_in_batches_in_one_transaction(mocker, tmp_path, fake_pool):
    # 1100 pages of one short chunk each
    _fake_reader(mocker, [f"page text {i}" for i in range(1100)])
    n = await ingest_pdf(tmp_path / "m.pdf", source_type="manual", title="Echo", car_id="toyota-echo-2001")
    assert n == 1100
    lock, delete, *inserts = fake_pool.calls
    assert lock[0] == "execute" and "pg_advisory_xact_lock(hashtextextended(" in lock[1]
    assert lock[2] == ("manual", "Echo", "toyota-echo-2001")
    assert delete[0] == "execute" and delete[1].startswith("DELETE FROM kb_passages WHERE source_type = $1")
    assert "car_id IS NOT DISTINCT FROM $3" in delete[1]
    assert delete[2] == ("manual", "Echo", "toyota-echo-2001")
    assert [m for m, *_ in inserts] == ["executemany"] * 3
    assert [len(c[2]) for c in inserts] == [500, 500, 100]
    assert fake_pool.transactions == ["commit"] and fake_pool.acquired == 0


async def test_ingest_pdf_row_shape(mocker, tmp_path, fake_pool):
    _fake_reader(mocker, ["hello"])
    await ingest_pdf(tmp_path / "b.pdf", source_type="book", title="Book")
    [(_, sql, rows)] = _batches(fake_pool)
    assert sql == "INSERT INTO kb_passages (source_type, car_id, title, page, content) VALUES ($1, $2, $3, $4, $5)"
    assert rows == (("book", None, "Book", 1, "hello"),)
    assert fake_pool.calls[1][2] == ("book", "Book", None)  # after the advisory lock


async def test_ingest_pdf_returns_zero_for_textless_pdf(mocker, tmp_path, fake_pool):
    _fake_reader(mocker, ["", None, "  \n "])
    assert await ingest_pdf(tmp_path / "s.pdf", source_type="book", title="Scan") == 0
    assert fake_pool.calls == [] and fake_pool.transactions == []


async def test_ingest_pdf_strips_nul_characters(mocker, tmp_path, fake_pool):
    _fake_reader(mocker, ["a\x00b"])
    await ingest_pdf(tmp_path / "n.pdf", source_type="book", title="Nul")
    assert _batches(fake_pool)[0][2][0][4] == "ab"


async def test_ingest_pdf_unreadable_pdf_never_opens_a_transaction(mocker, tmp_path, fake_pool):
    mocker.patch("app.services.kb.PdfReader", side_effect=ValueError("corrupt"))
    with pytest.raises(ValueError):
        await ingest_pdf(tmp_path / "bad.pdf", source_type="book", title="Bad")
    assert fake_pool.calls == [] and fake_pool.transactions == []


async def test_ingest_pdf_parses_before_any_connection_is_acquired(mocker, tmp_path, fake_pool):
    seen = []
    reader = _fake_reader(mocker, ["hello"])
    reader.return_value.pages[0].extract_text.side_effect = lambda: seen.append(fake_pool.acquired) or "hello"
    await ingest_pdf(tmp_path / "b.pdf", source_type="book", title="Book")
    assert seen == [0]


async def test_ingest_pdf_failed_second_batch_rolls_everything_back(mocker, tmp_path, fake_pool):
    fake_pool.error_on_executemany = 2
    _fake_reader(mocker, [f"page text {i}" for i in range(1100)])
    with pytest.raises(RuntimeError, match="batch failed"):
        await ingest_pdf(tmp_path / "m.pdf", source_type="manual", title="Echo", car_id="toyota-echo-2001")
    assert fake_pool.transactions == ["rollback"]  # the delete and batch 1 are undone with it
    assert fake_pool.acquired == 0
    assert len(_batches(fake_pool)) == 2  # batch 3 never ran
