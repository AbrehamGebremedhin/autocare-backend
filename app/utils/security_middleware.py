from fastapi import Request, Response
from app.core.config import get_settings
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable
import secrets
import os

DOCS_PATHS = {"/docs", "/docs/oauth2-redirect", "/redoc"}

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds security headers to all HTTP responses
    """
    
    def __init__(self, app, csp_nonce_header: str = "X-CSP-Nonce"):
        super().__init__(app)
        self.csp_nonce_header = csp_nonce_header
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate a nonce for CSP
        nonce = secrets.token_urlsafe(16)
        request.state.csp_nonce = nonce
        
        # Store request for CSP policy decisions
        self._current_request = request
        
        response = await call_next(request)
        
        # Determine CSP policy based on endpoint
        # Swagger UI / ReDoc pages need inline scripts and CDN assets, so only they skip the CSP (and only when ENABLE_DOCS serves them)
        is_docs_endpoint = get_settings().ENABLE_DOCS and request.url.path in DOCS_PATHS
        
        # Security headers
        headers = {
            # XSS protection
            "X-Content-Type-Options": "nosniff",
            "X-XSS-Protection": "1; mode=block",
            
            # HTTPS enforcement (if in production)
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains" if self._is_production() else "",
            
            # Referrer policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            
            # Permissions policy
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
            
            # Cache control for sensitive endpoints
            "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate" if self._is_sensitive_endpoint(request) else "public, max-age=300",
            
            # Custom nonce header for frontend
            self.csp_nonce_header: nonce
        }
        
        # Only add security headers for non-docs endpoints
        if not is_docs_endpoint:
            headers["Content-Security-Policy"] = self._get_csp_policy(nonce)
            headers["X-Frame-Options"] = "DENY"
        
        # A route's own Cache-Control (e.g. the SSE stream's no-cache) wins
        if "cache-control" in response.headers:
            del headers["Cache-Control"]

        # Add headers to response
        for header, value in headers.items():
            if value:  # Only add non-empty headers
                response.headers[header] = value
        
        return response
    
    def _get_csp_policy(self, nonce: str) -> str:
        """Generate Content Security Policy"""
        return (
            f"default-src 'self'; "
            f"script-src 'self' 'nonce-{nonce}' https://cdn.jsdelivr.net; "
            f"style-src 'self' 'nonce-{nonce}' https://fonts.googleapis.com; "
            f"font-src 'self' https://fonts.gstatic.com; "
            f"img-src 'self' data: https:; "
            f"connect-src 'self'; "
            f"frame-ancestors 'none'; "
            f"base-uri 'self'; "
            f"object-src 'none'"
        )
    
    def _is_production(self) -> bool:
        """Check if running in production"""
        return os.getenv("ENV", "development").lower() == "production"
    
    def _is_sensitive_endpoint(self, request: Request) -> bool:
        """Check if endpoint handles sensitive data"""
        sensitive_paths = ["/auth/", "/api/v1/auth/", "/security/", "/api/v1/sessions", "/api/v1/cars", "/api/v1/visuals/"]
        return any(sensitive_path in str(request.url.path) for sensitive_path in sensitive_paths)
