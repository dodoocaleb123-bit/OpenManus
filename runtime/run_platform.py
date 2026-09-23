import os

import uvicorn

if __name__ == "__main__":
    # Defaults keep local development on http://127.0.0.1:8000.
    # Containers/hosting platforms (e.g. Render) set HOST=0.0.0.0 and PORT.
    uvicorn.run(
        "app.server:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )
