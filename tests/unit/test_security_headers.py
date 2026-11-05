def test_security_headers_on_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    h = response.headers
    assert h["X-Content-Type-Options"] == "nosniff"
    assert h["X-XSS-Protection"] == "1; mode=block"
    assert h["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert h["Permissions-Policy"] == "geolocation=(), microphone=(), camera=()"
    assert h["X-Frame-Options"] == "DENY"
    assert h["Cache-Control"] == "public, max-age=300"
    assert "frame-ancestors 'none'" in h["Content-Security-Policy"]
    assert "Strict-Transport-Security" not in h  # only set when ENV=production


def test_csp_nonce_header_matches_csp(client):
    response = client.get("/health")

    nonce = response.headers["X-CSP-Nonce"]
    assert nonce
    assert f"'nonce-{nonce}'" in response.headers["Content-Security-Policy"]
