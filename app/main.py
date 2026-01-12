from fastapi import FastAPI
from app.api.router import api_router

app = FastAPI(title="AI Data Analyst")

app.include_router(api_router)
