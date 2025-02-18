import chromadb
import logging
import sys
import os

from llama_index.core.storage import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.elasticsearch import (
    ElasticsearchStore,
    AsyncDenseVectorStrategy,
)

from elasticsearch import Elasticsearch, AsyncElasticsearch
from rnd_chat_with_doc.modules.config.global_settings import get_config

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

config = get_config()


def get_chroma_storage_context():
    db = chromadb.PersistentClient(path=config.get("INDEX_STORAGE"))
    collection = db.get_or_create_collection("rnd_chroma_store")
    store = ChromaVectorStore(chroma_collection=collection)

    context = StorageContext.from_defaults(vector_store=store)

    return context


def get_chroma_collection():
    db = chromadb.PersistentClient(path=config.get("INDEX_STORAGE"))
    collection = db.get_or_create_collection("rnd_chroma_store")
    return collection


def get_es_client():
    es_client = Elasticsearch("http://localhost:9200")
    return es_client


def get_async_es_client():
    es_client = AsyncElasticsearch("http://localhost:9200", node_class="httpxasync")
    return es_client


def get_elastic_storage_context(
    index: str = "cc-document-embeddings", retrieval_strategy=AsyncDenseVectorStrategy()
):
    client = get_async_es_client()
 
    db = ElasticsearchStore(
        es_client=client,
        index_name=index,
        retrieval_strategy=retrieval_strategy,
    )

    storage_context = StorageContext.from_defaults(vector_store=db)
    return storage_context, db


if __name__ == "__main__":
    context = get_elastic_storage_context("jsjs")
    print(context)
