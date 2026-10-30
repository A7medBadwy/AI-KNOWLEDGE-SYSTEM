import hashlib
import json
import logging
import redis
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from openrouter.errors import (
    EdgeNetworkTimeoutResponseError,
    RequestTimeoutResponseError,
)
from config import (
    REDIS_URL,
    REDIS_TTL,
    QDRANT_PATH,
    QDRANT_COLLECTION,
    EMBEDDING_MODEL,
    MAX_ITERATIONS,
)
from agents import generate_answer, evaluate_answer

# Logging 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

class AIProviderError(RuntimeError):
    """Raised when the model provider cannot produce a usable answer."""


        # REDIS CACHE


redis_client=redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5,
)
CACHE_VERSION_KEY="knowledgeai:version"

def get_cache_version():
    version=redis_client.get(CACHE_VERSION_KEY)
    return version or "0"


def create_cache_key(question):
    version=get_cache_version()
    normalized_question=" ".join(question.lower().split())
    raw_key=f"2:{version}:{normalized_question}"
    
    hashed_key = hashlib.sha256(
        raw_key.encode("utf-8")
    ).hexdigest()
    
    return f"knowledgeai:answer:{hashed_key}"


def get_cached_answer(question):
    key=create_cache_key(question)
    cached_data=redis_client.get(key)
    if cached_data:
        logger.info("Cache hit")
        return json.loads(cached_data)
    logger.info("Cache miss")
    return None
    
def save_cached_answer(question, result):
    key = create_cache_key(question)

    redis_client.setex(
        key,
        REDIS_TTL,
        json.dumps(result, ensure_ascii=False),
    )
    

            # EMBEDDING & QDRANT VDBS

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

QDRANT_PATH.mkdir(parents=True, exist_ok=True)

qdrant_client= QdrantClient(path=str(QDRANT_PATH))


def get_vector_store():
    if qdrant_client.collection_exists(QDRANT_COLLECTION):
        return  QdrantVectorStore(
            client=qdrant_client,
            collection_name=QDRANT_COLLECTION,
            embedding=embeddings
                )        
    return None
vector_store= get_vector_store()

def add_document(documents):
    global vector_store
    
    if not documents:
        raise ValueError("No Documents were provided ")
    if vector_store is None :
        if not qdrant_client.collection_exists(QDRANT_COLLECTION):
            vector_size= len(embeddings.embed_query("test"))
            qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                ),
            )
        vector_store= QdrantVectorStore(
            client=qdrant_client,
            collection_name=QDRANT_COLLECTION,
            embedding=embeddings
        )
    vector_store.add_documents(documents)
    redis_client.incr(CACHE_VERSION_KEY)
    logger.info(
        "Added %s document chunks to Qdrant",
        len(documents),
    )
    
    
      # LCEL RETRIVER CHAIN
      
def retrieve_documents(question):
    if vector_store is None:
        return []
    retriever=vector_store.as_retriever(
        search_kwargs={"k":4}
    )
    return retriever.invoke(question)

retriever_chain=RunnableLambda(retrieve_documents)

def format_context(documents):
    if not documents :
        return "NONE RELEVANT KNOWLEDGE "
    
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n"
        f"Content: {doc.page_content}"
        for doc in documents) 

         # GENERATOR EVALUATOR :
def delete_document_chunks(filename):
    global vector_store

    if not qdrant_client.collection_exists(QDRANT_COLLECTION):
        return 0

    if vector_store is None:
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=QDRANT_COLLECTION,
            embedding=embeddings
        )

    result = vector_store.client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="metadata.filename",
                    match=MatchValue(value=filename)
                )
            ]
        ),
        wait=True
    )

    redis_client.incr(CACHE_VERSION_KEY)

    logger.info("Deleted chunks for %s", filename)

    return result

def answer_question(question):
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")
    question = question.strip()
    cached_result = get_cached_answer(question)
     
    if cached_result is not None:
        cached_result["cached"] = True
        return cached_result
    documents = retriever_chain.invoke(question)
    context = format_context(documents)
    
    feedback = ""
    answer = ""
    evaluation = {}
    evaluation_unverified = False
    for iteration in range(1, MAX_ITERATIONS + 1):

        logger.info(
            "Generator iteration %s/%s",
            iteration,
            MAX_ITERATIONS,
        )
        # GENERATION
        try:
            answer = generate_answer(
                question=question,
                context=context,
                feedback=feedback,
            )
        except (
            EdgeNetworkTimeoutResponseError,
            RequestTimeoutResponseError,
        ) as error:
            raise TimeoutError(
                "The AI provider timed out while generating an answer."
            ) from error
        if not answer or not answer.strip():
            raise AIProviderError(
                "The AI provider returned an empty answer. Try again or "
                "select a specific model in your OpenRouter settings."
            )
        # EVALUATION
        try:
            evaluation = evaluate_answer(
                question=question,
                context=context,
                answer=answer,
            )
        except (
            EdgeNetworkTimeoutResponseError,
            RequestTimeoutResponseError,
        ):
            evaluation_unverified = True
            evaluation = {
                "decision": "REVISE",
                "feedback": "The evaluator timed out; review this answer.",
                "issues": ["Evaluator timed out; answer is unverified."],
            }
            logger.warning("Evaluator timed out; returning an unverified answer")
        except ValueError:
            evaluation_unverified = True
            evaluation = {
                "decision": "REVISE",
                "feedback": "The evaluator returned an invalid response.",
                "issues": ["Evaluator response could not be validated."],
            }
            logger.warning("Evaluator returned an invalid response")

        logger.info(
            "Evaluator decision: %s",
            evaluation["decision"],
        )

        if evaluation["decision"] == "ACCEPT":
            break
        feedback = evaluation.get(
            "feedback",
            "Please improve the answer.",
        )
        
    accepted = evaluation.get("decision") == "ACCEPT"
        
    result = {
        "question": question,
        "answer": answer,
        "accepted": accepted,
        "iterations": iteration,
        "evaluation": evaluation,
        "cached": False,
            }
        
    if evaluation_unverified:
        result["warning"] = (
            "The evaluator could not validate this answer. It is unverified; "
            "please review it against your sources."
        )
    elif not accepted:
        result["warning"] = (
            "The answer was not validated. Please review it against your sources."
        )
        
    if accepted:
        save_cached_answer(question, result)

    return result


# TEST 

print("Testing Redis...")
print("Redis:", redis_client.ping())

print("\nTesting Qdrant...")
print("Qdrant collections:", qdrant_client.get_collections())

print("\nTesting Embedding Model...")
vector = embeddings.embed_query(
    "KnowledgeAI is an AI knowledge platform."
)

print("Embedding dimension:", len(vector))

print("\nTesting Retrieval Chain...")
documents = retriever_chain.invoke(
    "What is KnowledgeAI?"
)

print("Retrieved documents:", len(documents))

print("\nAll component tests completed!")
