"""
WSGI entrypoint for production deployment
"""
from api import app

# For gunicorn compatibility, we need to ensure the app is properly configured
application = app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
