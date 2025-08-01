from fastapi import Request
from fastapi.responses import HTMLResponse

async def custom_swagger_ui_html(request: Request) -> HTMLResponse:
    html_content = '''<!DOCTYPE html>
<html>
<head>
    <title>AutoCare API - Documentation</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui.css" />
    <style>
        html { box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; }
        *, *:before, *:after { box-sizing: inherit; }
        body { margin:0; background: #fafafa; }
        .loading { text-align: center; padding: 50px; font-family: Arial, sans-serif; }
        .error { text-align: center; padding: 50px; font-family: Arial, sans-serif; color: #d32f2f; background: #ffebee; border: 1px solid #f8bbd9; border-radius: 4px; margin: 20px; }
    </style>
</head>
<body>
    <div id="swagger-ui">
        <div class="loading">
            <h2>Loading AutoCare API Documentation...</h2>
            <p>Please wait while we load the API documentation.</p>
        </div>
    </div>
    <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-bundle.js"></script>
    <script>
        function showError(message) {
            document.getElementById('swagger-ui').innerHTML = '<div class="error"><h2>Documentation Load Error</h2><p>' + message + '</p></div>';
        }
        
        function initSwagger() {
            try {
                const ui = SwaggerUIBundle({
                    url: '/openapi.json',
                    dom_id: '#swagger-ui',
                    deepLinking: true,
                    presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.presets.standalone],
                    plugins: [SwaggerUIBundle.plugins.DownloadUrl],
                    layout: "StandaloneLayout",
                    onComplete: function() { console.log('Swagger UI loaded successfully!'); },
                    onFailure: function(err) { console.error('Swagger UI failed:', err); showError('Failed to load API documentation.'); }
                });
            } catch (error) {
                console.error('Error during Swagger UI setup:', error);
                showError('Error initializing documentation: ' + error.message);
            }
        }
        
        if (typeof SwaggerUIBundle !== 'undefined') {
            initSwagger();
        } else {
            window.onload = function() {
                if (typeof SwaggerUIBundle !== 'undefined') {
                    initSwagger();
                } else {
                    showError('Failed to load Swagger UI library.');
                }
            };
        }
    </script>
</body>
</html>'''
    return HTMLResponse(content=html_content)
