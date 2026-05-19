from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.retriever import search
from core.ingestor import ingest_file


app = FastAPI(title="RAG Demo API", version="1.0.0")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query")
    verbose: bool = Field(default=False, description="Print pipeline details to server logs")


class IngestRequest(BaseModel):
    file_path: str = Field(..., min_length=1, description="Absolute or relative path to text file")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/query")
async def query_rag(payload: QueryRequest) -> dict:
    try:
        return await search(payload.query, verbose=payload.verbose)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc


@app.post("/ingest")
async def ingest(payload: IngestRequest) -> dict:
    try:
        doc_id = await ingest_file(payload.file_path)
        return {"status": "ok", "doc_id": doc_id, "file_path": payload.file_path}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {payload.file_path}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}") from exc
 