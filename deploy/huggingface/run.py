import os

os.environ.setdefault("MODEL_URI", "/app/model")

import uvicorn

from api.app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
