from llama_index.llms.openai import OpenAI
from openai import OpenAI as DefOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from llama_index.embeddings.nvidia.base import NVIDIAEmbedding


def get_llm():
    llm = OpenAI(
        temperature=0.1,
        api_base="http://localhost:5000/v1",
        api_key="test",
        timeout=900,
        max_retries=0,
    )

    return llm


def get_nim_llm():
    key = os.getenv("NVIDIA_KEY")
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=key,
    )
    return client


def get_embedding_model():
    emb = OpenAIEmbedding(
        api_base="http://localhost:5000/v1/",
        api_key="test",
        max_retries=1,
        timeout=900,
    )
    return emb


def get_nim_embedding_model():
    key = os.getenv("NVIDIA_KEY")

    emb = NVIDIAEmbedding(
        base_url="https://ai.api.nvidia.com/v1/retrieval/snowflake/arctic-embed-l",
        model="snowflake/arctic-embed-l",
        api_key=key,
    )
    return emb


def test_get_embedding_model():
    emb = get_embedding_model()
    r = emb.get_text_embedding("Hi")
    print(r)


# Custom embedding model function to connect to your local LLM
async def async_get_local_embeddings(text):
    emb = get_embedding_model()
    r = await emb.aget_text_embedding(text)


def get_local_embeddings(text):
    emb = get_embedding_model()
    return emb.get_text_embedding(text)


def get_local_embedding_oai(text: str):
    text = text.replace("\n", " ")
    client = DefOpenAI(base_url="http://localhost:5000/v1", api_key="lm-studio")
    return (
        client.embeddings.create(
            input=[text], model="nomic-ai/nomic-embed-text-v1.5-GGUF"
        )
        .data[0]
        .embedding
    )


def get_local_openai_embeddind():
    client = DefOpenAI(base_url="http://localhost:5000/v1", api_key="lm-studio")
    return client


async def async_get_llm():
    try:
        llm = OpenAI(
            temperature=0,
            api_base="http://localhost:5000/v1",
            api_key="test",
            timeout=30,
            max_retries=3,
        )
        return llm
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        raise


async def async_get_embedding_model():
    try:
        emb = OpenAIEmbedding(
            api_base="http://localhost:5000/v1",
            api_key="test",
        )
        return emb
    except Exception as e:
        print(f"Failed to initialize embedding model: {e}")
        raise


if __name__ == "__main__":
    emb = test_get_embedding_model()
    print(emb)
