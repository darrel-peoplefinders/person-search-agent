"""
Startup script for production deployment
"""
import os
import sys

def main():
    """Main entrypoint for the application"""
    port = int(os.environ.get("PORT", 8080))
    
    # Import the app after setting up the environment
    from api import app
    
    # Run with uvicorn for async compatibility
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        workers=1,  # Single worker for Cloud Run
        loop="asyncio"
    )

if __name__ == "__main__":
    main()
