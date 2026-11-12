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
        {"url": "https://files.test/2010_toyota_echo.pdf", "title": "2010 Echo"},
    ]
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"web": {"results": hits}}))
    urls = await manual_search.search_manual_urls("Toyota", "Echo", 2001, 5, transport=transport)
    assert urls == ["https://files.test/toyota-echo-2001-owners-manual.pdf", "https://blog.test/echo-tips"]


async def test_search_without_key_or_on_failure_is_empty(brave_key, mocker):
    assert await manual_search.search_manual_urls("Toyota", "Echo", 2001, 0) == []
    boom = httpx.MockTransport(lambda r: httpx.Response(500))
    assert await manual_search.search_manual_urls("Toyota", "Echo", 2001, 5, transport=boom) == []
    mocker.patch("app.services.manual_search.get_settings", return_value=mocker.Mock(BRAVE_API_KEY=""))
    assert await manual_search.search_manual_urls("Toyota", "Echo", 2001, 5) == []


async def test_download_accepts_a_pdf_about_the_model(tmp_path, brave_key):
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=_pdf("Toyota Echo owners manual")))
    assert await manual_search.download_manual("https://a.test/m.pdf", tmp_path / "m.pdf", ["Echo"], transport=transport)


async def test_download_rejects_other_models_non_pdfs_and_oversize(tmp_path, brave_key):
    def serve(content):
        return httpx.MockTransport(lambda r: httpx.Response(200, content=content))

    dest = tmp_path / "m.pdf"
    assert not await manual_search.download_manual("https://a.test/m.pdf", dest, ["Echo"], transport=serve(_pdf("Honda Civic")))
    assert not await manual_search.download_manual("https://a.test/m.pdf", dest, ["Echo"], transport=serve(b"<html>Echo</html>"))
    assert not await manual_search.download_manual("https://a.test/m.pdf", dest, ["Echo"], transport=serve(b"%PDF-" + b"0" * 2_000_000))


async def test_download_refuses_private_hosts_and_private_redirects(tmp_path, brave_key):
    dest = tmp_path / "m.pdf"
    never = httpx.MockTransport(lambda r: pytest.fail("fetched a private host"))
    assert not await manual_search.download_manual("http://internal.test/m.pdf", dest, ["Echo"], transport=never)
    redirect = httpx.MockTransport(lambda r: httpx.Response(302, headers={"location": "http://internal.test/m.pdf"}))
    assert not await manual_search.download_manual("https://a.test/m.pdf", dest, ["Echo"], transport=redirect)


async def test_search_and_check_accept_sibling_names(brave_key, tmp_path):
    hits = [{"url": "https://files.test/echo-owners-manual.pdf", "title": "Echo manual"}]
    queries = []

    def handler(request):
        queries.append(request.url.params["q"])
        return httpx.Response(200, json={"web": {"results": hits}})

    urls = await manual_search.search_manual_urls("Toyota", "Platz", 2001, 5, ["Echo"], transport=httpx.MockTransport(handler))
    assert urls == ["https://files.test/echo-owners-manual.pdf"] and any("Echo" in q for q in queries)
    pdf = httpx.MockTransport(lambda r: httpx.Response(200, content=_pdf("Toyota Echo owners manual")))
    assert not await manual_search.download_manual("https://a.test/m.pdf", tmp_path / "m.pdf", ["Platz", "Echo"], 2001, transport=pdf)
    dated = httpx.MockTransport(lambda r: httpx.Response(200, content=_pdf("Toyota Echo 2001 owners manual")))
    assert await manual_search.download_manual("https://a.test/m.pdf", tmp_path / "m.pdf", ["Platz", "Echo"], 2001, transport=dated)


async def test_sibling_names_cleans_the_llm_answer_and_survives_failure(mocker):
    def reply(content):
        return mocker.Mock(choices=[mocker.Mock(message=mocker.Mock(content=content))])

    create = mocker.AsyncMock(return_value=reply('{"names": ["Echo", "Platz", "Yaris", "bad<script>", 5]}'))
    mocker.patch("app.services.manual_search.llm.get_client", return_value=mocker.Mock(chat=mocker.Mock(completions=mocker.Mock(create=create))))
    assert await manual_search.sibling_names("Toyota", "Platz", 2001) == ["Echo", "Yaris"]
    create.side_effect = RuntimeError("down")
    assert await manual_search.sibling_names("Toyota", "Platz", 2001) == []


async def test_ingest_indexes_web_manuals_after_startmycar(mocker):
    mocker.patch("app.CRUD.car_crud.get_settings", return_value=mocker.Mock(MANUAL_MAX_SOURCES=3, BRAVE_API_KEY="k"))
    mocker.patch("app.CRUD.car_crud.manual_search.sibling_names", return_value=[])
    mocker.patch("app.CRUD.car_crud.FetchCarDataService.scrape_links", return_value=None)
    mocker.patch("app.CRUD.car_crud.manual_search.search_manual_urls", return_value=["https://a.test/1.pdf", "https://b.test/2.pdf"])
    mocker.patch("app.CRUD.car_crud.manual_search.download_manual", side_effect=[False, True])
    ingest = mocker.patch("app.CRUD.car_crud.kb.ingest_pdf", return_value=10)
    update = mocker.patch("app.CRUD.car_crud.CarCRUD.update")
    await ingest_car_manual(CAR)
    assert ingest.call_args.kwargs["title"] == "Toyota Echo 2001 manual (b.test)"
    assert update.call_args.args[1] == {"kb_status": "ready", "kb_passage_count": 10}


async def test_fetch_pdf_keeps_a_public_pdf(tmp_path, brave_key):
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=_pdf("anything")))
    dest = tmp_path / "m.pdf"
    assert await manual_search.fetch_pdf("https://a.test/m.pdf", dest, transport=transport)
    assert dest.read_bytes().startswith(b"%PDF-")


async def test_fetch_pdf_refuses_a_private_address(tmp_path, brave_key):
    never = httpx.MockTransport(lambda r: pytest.fail("fetched a private host"))
    assert not await manual_search.fetch_pdf("http://internal.test/m.pdf", tmp_path / "m.pdf", transport=never)


async def test_fetch_pdf_refuses_a_body_that_is_not_a_pdf(tmp_path, brave_key):
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=b"<html>"))
    assert not await manual_search.fetch_pdf("https://a.test/m.pdf", tmp_path / "m.pdf", transport=transport)


async def test_fetch_pdf_stops_at_the_size_cap(tmp_path, brave_key):  # brave_key sets MANUAL_MAX_MB=1
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=b"%PDF-" + b"0" * 2_000_000))
    assert not await manual_search.fetch_pdf("https://a.test/m.pdf", tmp_path / "m.pdf", transport=transport)


async def test_fetch_pdf_rechecks_the_address_after_a_redirect(tmp_path, brave_key):
    def serve(request):
        if request.url.host == "a.test":
            return httpx.Response(302, headers={"location": "http://internal.test/m.pdf"})
        pytest.fail("followed a redirect to a private host")

    assert not await manual_search.fetch_pdf("https://a.test/m.pdf", tmp_path / "m.pdf", transport=httpx.MockTransport(serve))


def test_build_url_quotes_make_and_model():
    from app.services.fetch_car_data_service import FetchCarDataService

    urls = FetchCarDataService().build_url("a/b?x", "c#d", 2001)
    assert urls["Owner_Manual"] == "https://www.startmycar.com/a%2Fb%3Fx/c%23d/info/manuals/2001"
