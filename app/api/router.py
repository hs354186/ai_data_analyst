# app/api/router.py
from fastapi import APIRouter
from app.api.routes import upload, semantics, rag

api_router = APIRouter()
api_router.include_router(rag.router, tags=["rag"])
api_router.include_router(upload.router)
api_router.include_router(semantics.router)
api_router.include_router(rag.router)
