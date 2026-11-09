import io

import httpx
import pytest

from app.CRUD.car_crud import ingest_car_manual
from app.services import manual_search

CAR = {"id": "toyota-echo-2001", "make": "Toyota", "model": "Echo", "year": 2001}


def _pdf(text: str) -> bytes:
    # A minimal hand-written PDF whose one page shows `text`.
    body = f"BT /F1 12 Tf 10 100 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(body)).encode() + b" >>\nstream\n" + body + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = io.BytesIO(), []
    out.write(b"%PDF-1.4\n")
    for i, o in enumerate(objs, 1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode() + o + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode())
    return out.getvalue()


@pytest.fixture(autouse=True)
def public_hosts(mocker):
    async def is_public(url):
        return "internal.test" not in url

    mocker.patch.object(manual_search, "_is_public", is_public)
    mocker.patch.object(manual_search, "BRAVE_PAUSE_SECONDS", 0)


@pytest.fixture
def brave_key(mocker):
    mocker.patch("app.services.manual_search.get_settings", return_value=mocker.Mock(BRAVE_API_KEY="k", MANUAL_MAX_MB=1))


async def test_search_ranks_pdf_hits_first_and_drops_other_models(brave_key):
    hits = [
        {"url": "https://blog.test/echo-tips", "title": "Echo tips"},
        {"url": "https://files.test/toyota-echo-2001-owners-manual.pdf", "title": "Toyota Echo 2001 manual"},
        {"url": "https://other.test/x.pdf", "title": "unrelated"},
    ]
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"web": {"results": hits}}))
    urls = await manual_search.search_manual_urls("Toyota", "Echo", 2001, 5, transport)
    assert urls == ["https://files.test/toyota-echo-2001-owners-manual.pdf", "https://blog.test/echo-tips"]


async def test_search_without_key_or_on_failure_is_empty(brave_key, mocker):
    assert await manual_search.search_manual_urls("Toyota", "Echo", 2001, 0) == []
    boom = httpx.MockTransport(lambda r: httpx.Response(500))
    assert await manual_search.search_manual_urls("Toyota", "Echo", 2001, 5, boom) == []
    mocker.patch("app.services.manual_search.get_settings", return_value=mocker.Mock(BRAVE_API_KEY=""))
    assert await manual_search.search_manual_urls("Toyota", "Echo", 2001, 5) == []


async def test_download_accepts_a_pdf_about_the_model(tmp_path, brave_key):
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=_pdf("Toyota Echo owners manual")))
    assert await manual_search.download_manual("https://a.test/m.pdf", tmp_path / "m.pdf", "Echo", transport)


async def test_download_rejects_other_models_non_pdfs_and_oversize(tmp_path, brave_key):
    def serve(content):
        return httpx.MockTransport(lambda r: httpx.Response(200, content=content))

    dest = tmp_path / "m.pdf"
    assert not await manual_search.download_manual("https://a.test/m.pdf", dest, "Echo", serve(_pdf("Honda Civic")))
    assert not await manual_search.download_manual("https://a.test/m.pdf", dest, "Echo", serve(b"<html>Echo</html>"))
    assert not await manual_search.download_manual("https://a.test/m.pdf", dest, "Echo", serve(b"%PDF-" + b"0" * 2_000_000))


async def test_download_refuses_private_hosts_and_private_redirects(tmp_path, brave_key):
    dest = tmp_path / "m.pdf"
    never = httpx.MockTransport(lambda r: pytest.fail("fetched a private host"))
    assert not await manual_search.download_manual("http://internal.test/m.pdf", dest, "Echo", never)
    redirect = httpx.MockTransport(lambda r: httpx.Response(302, headers={"location": "http://internal.test/m.pdf"}))
    assert not await manual_search.download_manual("https://a.test/m.pdf", dest, "Echo", redirect)


async def test_ingest_indexes_web_manuals_after_startmycar(mocker):
    mocker.patch("app.CRUD.car_crud.FetchCarDataService.scrape_links", return_value=None)
    mocker.patch("app.CRUD.car_crud.manual_search.search_manual_urls", return_value=["https://a.test/1.pdf", "https://b.test/2.pdf"])
    mocker.patch("app.CRUD.car_crud.manual_search.download_manual", side_effect=[False, True])
    ingest = mocker.patch("app.CRUD.car_crud.kb.ingest_pdf", return_value=10)
    update = mocker.patch("app.CRUD.car_crud.CarCRUD.update")
    await ingest_car_manual(CAR)
    assert ingest.call_args.kwargs["title"] == "Toyota Echo 2001 manual (b.test)"
    assert update.call_args.args[1] == {"kb_status": "ready", "kb_passage_count": 10}
