from fastapi import FastAPI, UploadFile, File, Form, HTTPException, logger
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pathlib import Path
from tempfile import NamedTemporaryFile
from ingestion import (
    MAX_FILE_SIZE,
    WHISPER_MODELS,
    chunk_documents,
    load_file,
    transcribe_audio,
)
from work_flow import (
    AIProviderError,
    add_document,
    answer_question,
    delete_document_chunks,
)
from pydantic import BaseModel
from datetime import datetime
app = FastAPI(
    title="KnowledgeAI API",
    description="Evaluator-Generator AI Knowledge Platform"
)

app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent / "static"),
    name="static",
)

@app.get("/")
async def home():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
    
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "application": "KnowledgeAI"
    }

@app.post("/upload")
async def upload_file(file:UploadFile=File(...)):
    upload_dir=Path("uploaded_files")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path= upload_dir/Path(file.filename).name
    
    try:
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty"
            )

        file_path.write_bytes(content)
        documents = load_file(str(file_path))
        chunks = chunk_documents([documents])

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from file"
                                )

        for chunk in chunks:
            chunk.metadata["filename"] = file_path.name

        add_document(chunks)

        return {
            "filename": file.filename,
            "status": "processed",
            "chunks_count": len(chunks),
            "message": "File uploaded and stored in Qdrant successfully"
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        await file.close()

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QuestionRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:
        result = await run_in_threadpool(answer_question, request.question)

        return {
            "question": request.question,
            "answer": result["answer"],
            "accepted": result["accepted"],
            "iterations": result["iterations"],
            "evaluation": result["evaluation"],
            "warning": result.get("warning"),
        }

    except TimeoutError as e:
        raise HTTPException(
            status_code=504,
            detail=str(e),
        )

    except AIProviderError as e:
        raise HTTPException(
            status_code=502,
            detail=str(e),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/ask-audio")
async def ask_audio_question(
    file: UploadFile = File(...),
    model_size: str = Form("base"),
):
    extension = Path(file.filename or "").suffix.lower()
    supported_audio = {
        ".wav", ".mp3", ".m4a", ".webm", ".mp4", ".mpeg", ".mpga",
        ".ogg", ".flac",
    }
    if extension not in supported_audio:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format. Use: {', '.join(sorted(supported_audio))}.",
        )
    if model_size not in WHISPER_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported Whisper model. Choose one of: {', '.join(sorted(WHISPER_MODELS))}.",
        )

    temp_path = None
    try:
        content = await file.read(MAX_FILE_SIZE + 1)
        if not content:
            raise HTTPException(status_code=400, detail="Audio file is empty.")
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Audio file exceeds the maximum size of 50MB.",
            )

        with NamedTemporaryFile(
            mode="wb",
            suffix=extension,
            delete=False,
        ) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        transcript_doc = await run_in_threadpool(
            transcribe_audio,
            temp_path,
            model_size,
        )
        result = await run_in_threadpool(
            answer_question,
            transcript_doc.page_content,
        )
        return {
            "question": transcript_doc.page_content,
            "transcript": transcript_doc.page_content,
            "model_size": model_size,
            "answer": result["answer"],
            "accepted": result["accepted"],
            "iterations": result["iterations"],
            "evaluation": result["evaluation"],
            "warning": result.get("warning"),
        }
    except HTTPException:
        raise
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except AIProviderError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

@app.get("/documents")
async def get_documents():

    upload_dir = Path("uploaded_files")

    if not upload_dir.exists():
        return {
            "documents": [],
            "total": 0
        }

    documents = []

    for file_path in upload_dir.iterdir():

        if file_path.is_file():
            documents.append({
                "filename": file_path.name,
                "extension": file_path.suffix,
                "size_kb": round(file_path.stat().st_size / 1024, 2),
                "uploaded_at": datetime.fromtimestamp(
                    file_path.stat().st_mtime
                ).isoformat()
            })

    return {
        "documents": documents,
        "total": len(documents)
    }
@app.delete("/documents/{filename}")
async def delete_document(filename: str):

    upload_dir = Path("uploaded_files")
    file_path = upload_dir / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    try:
        delete_document_chunks(filename)
        file_path.unlink()

        return {
            "message": f"{filename} deleted successfully"
        }

    except Exception as e:
        logger.exception("Failed to delete document")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )