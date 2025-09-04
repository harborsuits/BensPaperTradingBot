# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import decisions

app = FastAPI(title="BenBot API")

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the decisions router
app.include_router(decisions.router)

# Root endpoint for health check
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Decision Trace API is running"}

# For running with uvicorn directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

