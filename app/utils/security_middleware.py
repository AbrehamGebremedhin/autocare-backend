from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable
import secrets
import os

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
        is_docs_endpoint = '/docs' in str(request.url.path) or '/redoc' in str(request.url.path)
        
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
            headers["Content-Security-Policy"] = self._get_csp_policy(nonce, is_docs_endpoint)
            headers["X-Frame-Options"] = "DENY"
        
        # Add headers to response
        for header, value in headers.items():
            if value:  # Only add non-empty headers
                response.headers[header] = value
        
        return response
    
    def _get_csp_policy(self, nonce: str, is_docs_endpoint: bool = False) -> str:
        """Generate Content Security Policy"""
        
        if is_docs_endpoint:
            # Very relaxed CSP for documentation pages to allow all Swagger UI resources
            return (
                f"default-src *; "
                f"script-src * 'unsafe-inline' 'unsafe-eval'; "
                f"style-src * 'unsafe-inline'; "
                f"font-src *; "
                f"img-src * data:; "
                f"connect-src *; "
                f"frame-src *; "
                f"object-src *"
            )
        else:
            # Strict CSP for other endpoints
            return (
                f"default-src 'self'; "
                f"script-src 'self' 'nonce-{nonce}' 'unsafe-inline' https://cdn.jsdelivr.net; "
                f"style-src 'self' 'nonce-{nonce}' 'unsafe-inline' https://fonts.googleapis.com; "
                f"font-src 'self' https://fonts.gstatic.com; "
                f"img-src 'self' data: https:; "
                f"connect-src 'self' wss: ws:; "
                f"frame-ancestors 'none'; "
                f"base-uri 'self'; "
                f"object-src 'none'"
            )
    
    def _is_production(self) -> bool:
        """Check if running in production"""
        return os.getenv("ENV", "development").lower() == "production"
    
    def _is_sensitive_endpoint(self, request: Request) -> bool:
        """Check if endpoint handles sensitive data"""
        sensitive_paths = ["/auth/", "/api/v1/auth/", "/security/"]
        return any(sensitive_path in str(request.url.path) for sensitive_path in sensitive_paths)
