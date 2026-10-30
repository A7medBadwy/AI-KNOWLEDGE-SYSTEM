from functools import lru_cache
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse
import requests
import wikipedia
from faster_whisper import WhisperModel
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from pptx import Presentation
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".md", ".py", ".ipynb",
    ".js", ".ts", ".html", ".css", ".json", ".csv",
    ".pptx", ".ppt",
}

MAX_FILE_SIZE = 50 * 1024 * 1024
REQUEST_TIMEOUT = 25
WHISPER_MODELS = {"tiny", "base", "small", "medium", "large-v3"}
_whisper_lock = Lock()


@lru_cache(maxsize=1)
def _load_whisper_model(model_size):
    return WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
    )

def extract_text_from_file(file_path):
    "extract text from file"
    path= Path(file_path)
    extention = path.suffix.lower()
    if not path.is_file():
        raise FileNotFoundError(f"File {file_path} does not exist.")
    if path.stat().st_size > MAX_FILE_SIZE:
        raise ValueError(f"File {file_path} exceeds the maximum size of 50MB.")
    if extention not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {extention}. Supported extensions are: {', '.join(SUPPORTED_EXTENSIONS)}")
    if extention == ".pdf":
        reader=PdfReader(str(path))
        text="\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif extention == ".docx":
        doc=DocxDocument(str(path))
        text="\n".join(paragraph.text for paragraph in doc.paragraphs)
    elif extention in {".pptx", ".ppt"}:
        presentation=Presentation(str(path))
        slides_text=[]
        for slide in presentation.slides:
            for shape in slide.shapes:
                if shape.has_text_frame:
                    slides_text.append(shape.text)
        text="\n".join(slides_text)
    elif extention in {".txt", ".md", ".py", ".ipynb", ".js", ".ts", ".html", ".css", ".json", ".csv"}:
        with open(str(path), "r", encoding="utf-8") as f:
            text=f.read()
    else:
        raise ValueError(f"Unsupported file extension: {extention}. Supported extensions are: {', '.join(SUPPORTED_EXTENSIONS)}")
    if not text.strip():
        raise ValueError(f"No text could be extracted from the file {file_path}.") 
    return text 
def load_file(file_path):
    path = Path(file_path)
    text = extract_text_from_file(path)
    return Document(page_content=text, metadata={"source": path.name , "source_type": "file","file_type": path.suffix.lower()})
def load_web_page(url):
    parsed=urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"Invalid URL scheme: {parsed.scheme}. Only http and https are supported.")
    hostname=parsed.hostname.lower()
    if (
        hostname == "localhost"
        or hostname.endswith(".localhost")
        or hostname.endswith(".local")
        or hostname in {"127.0.0.1", "::1"}
        or hostname.startswith(("10.", "192.168.", "169.254."))
        or hostname == "0.0.0.0"
        or hostname.startswith("172.")
        and hostname.split(".")[1].isdigit()
        and 16 <= int(hostname.split(".")[1]) <= 31
    ):
        raise ValueError("Local and private network URLs are not allowed.")
    response=requests.get(url, timeout=REQUEST_TIMEOUT,headers={"User-Agent": "KnowledgeAI/1.0"})
    response.raise_for_status()
    content_type=response.headers.get("Content-Type", "").lower()
    if "text/html" not in content_type:
        raise ValueError(f"URL does not point to an HTML page. Content-Type: {content_type}")
    soup=BeautifulSoup(response.text, "html.parser")
    for element in soup(["script", "style", "nav", "footer", "header"]):    
        element.decompose()
        
    title = soup.title.get_text(" ", strip=True) if soup.title else url
    text = soup.get_text(separator="\n", strip=True)
    if not text:
        raise ValueError("No readable text found on the webpage.")
    return Document(
        page_content=text,
        metadata={
            "source": url,
            "title": title,
            "source_type": "webpage",
        },
    )
def load_wikipedia_page(title, language="en"):
    "Load a Wikipedia article by title."
    wikipedia.set_lang(language)

    page = wikipedia.page(title, auto_suggest=False)

    return Document(
        page_content=page.content,
        metadata={
            "source": page.url,
            "title": page.title,
            "source_type": "wikipedia",
        },
    )
    
def transcribe_audio(file_path, model_size="base"):
    "Transcribe audio with a selectable, cached Whisper model."
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File {file_path} does not exist.")
    if path.suffix.lower() not in {
        ".wav", ".mp3", ".m4a", ".webm", ".mp4", ".mpeg", ".mpga",
        ".ogg", ".flac",
    }:
        raise ValueError("Unsupported audio format.")
    if path.stat().st_size > MAX_FILE_SIZE:
        raise ValueError(f"File {file_path} exceeds the maximum size of 50MB.")
    if model_size not in WHISPER_MODELS:
        raise ValueError(
            f"Unsupported Whisper model. Choose one of: {', '.join(sorted(WHISPER_MODELS))}."
        )

    model = _load_whisper_model(model_size)
    with _whisper_lock:
        segments, _ = model.transcribe(str(path),language="en")
        text = " ".join(
            segment.text.strip() for segment in segments
        ).strip()
    
    if not text:
        raise ValueError("No speech was detected in the audio.")
    return Document(
        page_content=text,
        metadata={
            "source": path.name,
            "source_type": "audio",
            "transcription_model": model_size,
        },
    )
    
    
def chunk_documents(
    documents,
    chunk_size=1000,
    chunk_overlap=200,
):
    """Split documents into searchable chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    return chunks

"""

# Create a sample text file
sample_path = Path("sample.txt")

sample_path.write_text(
    
    KnowledgeAI is an AI-powered knowledge platform.
    It uses Retrieval Augmented Generation (RAG).
    The Generator creates answers using retrieved context.
    The Evaluator checks accuracy, relevance, and completeness.
    Redis is used for caching responses.
    Qdrant stores vector embeddings.
    ,
    encoding="utf-8"
)

# Extract text
document = load_file(sample_path)

print("Extracted text:")
print(document.page_content)

# Split text into chunks
chunks = chunk_documents(
    [document],
    chunk_size=100,
    chunk_overlap=20
)

print("\nNumber of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i + 1}:")
    print(chunk.page_content)
    print("Metadata:", chunk.metadata)

"""