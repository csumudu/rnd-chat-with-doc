from llama_index.core.schema import TextNode
from llama_index.core import StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import (
    TextSplitter,
    TokenTextSplitter,
    SimpleFileNodeParser,
)
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.vector_stores.elasticsearch import AsyncDenseVectorStrategy

from rnd_chat_with_doc.modules.config.global_settings import get_config
from rnd_chat_with_doc.modules.manage.vector_storage_context_manager import (
    get_elastic_storage_context,
)
from rnd_chat_with_doc.modules.util.llm_manager import get_llm
from rnd_chat_with_doc.modules.util.logging_manager import log
from llama_index.core import Settings
import asyncio


def print_results(results):
    for rank, result in enumerate(results, 1):
        print(
            f"{rank}. title={result.metadata['page_label']} score={result.get_score()} text={result.get_text()}"
        )


def search(storage_context: StorageContext, nodes: list[TextNode], query: str):
    index = VectorStoreIndex(nodes, storage_context=storage_context)

    print(">>> Documents:")
    retriever = index.as_retriever()
    results = retriever.retrieve(query)
    print_results(results)

    print("\n>>> Answer:")
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    print(response)


def query(storage_context: StorageContext, query: str):
    # index = VectorStoreIndex.from_vector_store(vector_store=s)
    index = VectorStoreIndex(nodes=[], storage_context=storage_context)

    print(">>> Documents:")
    retriever = index.as_retriever()
    results = retriever.retrieve(query)
    print_results(results)

    print("\n>>> Answer:")
    query_engine = index.as_query_engine(stream=True)
    response = query_engine.query(query)
    print(response)


async def elk_demo():
    context, db = get_elastic_storage_context("llamaindex_book")

    dir = get_config().get("LARGE_DOC_STORAGE")
    log(f"Directory Path: {dir}")

    docs = SimpleDirectoryReader(input_dir=dir).load_data()
    pharser = SimpleFileNodeParser.from_defaults()
    nodes = pharser.get_nodes_from_documents(documents=docs)

    # search(context, nodes, "what is RouterQueryEngine")

    while True:
        q = input("Question :")
        if q != "quit":
            query(context, q)
        db.close()
        break



if __name__ == "__main__":
    # model = get_llm()
    # Settings.llm = model

    asyncio.run(elk_demo())
