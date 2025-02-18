from flask import Flask, request, Response, stream_with_context, jsonify
from flask_cors import CORS
from llama_index.core import (
    SummaryIndex,
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
)
import logging
import sys
from llama_index.core.chat_engine.types import ChatMode

import time
from llama_index.core import get_response_synthesizer

from rnd_chat_with_doc.modules.config.global_settings import get_config
from rnd_chat_with_doc.modules.manage.vector_storage_context_manager import (
    get_elastic_storage_context,
    get_es_client,
)
from rnd_chat_with_doc.modules.util.formatters import get_formatted_response
from rnd_chat_with_doc.modules.util.llama_index_utils import (
    set_llamaindex_logging,
    set_llm,
    set_llm_nvidia,
)
from rnd_chat_with_doc.modules.util.llm_manager import (
    async_get_llm,
    get_embedding_model,
    get_llm,
)
from llama_index.llms.openai import OpenAI

from rnd_chat_with_doc.modules.util.logging_manager import log
from llama_index.core.node_parser import (
    SimpleFileNodeParser,
    SimpleNodeParser,
    SentenceWindowNodeParser,
    SemanticSplitterNodeParser,
    SentenceSplitter,
)
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core import ChatPromptTemplate

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

set_llamaindex_logging()
set_llm()

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return "Hello Flask"


def node_callback(node):
    logger = logging.getLogger()
    logger.debug("Node retrieved: %s", node)


@app.route("/upload", methods=["POST"])
def upload():
    index_name = request.json["indexName"]
    is_override = request.json["isOverride"]
    loader_type = request.json["loader"]

    es_client = get_es_client()
    context, db = get_elastic_storage_context(index_name)

    if is_override and es_client.indices.exists(index=index_name):
        es_client.indices.delete(index=index_name)

    if es_client.indices.exists(index=index_name):
        print("Index Exist")
        return "Index already exist"
    else:
        dir = get_config().get("LARGE_DOC_STORAGE")
        log(f"Directory Path: {dir}")

        if loader_type == "web":
            docs = SimpleWebPageReader(html_to_text=True).load_data(
                ["http://localhost:8080/ConfigGuide.html"]
            )
            # splitter = SentenceSplitter()
            # splitter = SentenceWindowNodeParser.from_defaults(
            #     # how many sentences on either side to capture
            #     window_size=3,
            #     # the metadata key that holds the window of surrounding sentences
            #     window_metadata_key="window",
            #     # the metadata key that holds the original sentence
            #     original_text_metadata_key="original_sentence",
            # )

            emb = get_embedding_model()
            splitter = SemanticSplitterNodeParser(
                buffer_size=1, breakpoint_percentile_threshold=95, embed_model=emb
            )
            nodes = splitter.get_nodes_from_documents(documents=docs)
        elif loader_type == "dir":
            docs = SimpleDirectoryReader(
                input_files=[f"{dir}\config_guide.pdf"]
            ).load_data()
            pharser = SimpleFileNodeParser.from_defaults()
            nodes = pharser.get_nodes_from_documents(documents=docs)
        elif loader_type == "simple":  # This seems working well - BEST
            docs = SimpleDirectoryReader(input_dir=dir).load_data()
            pharser = SimpleNodeParser.from_defaults(
                chunk_size=500,
                chunk_overlap=20,
                paragraph_separator="\n\n",  # BEST CONFIG
            )
            nodes = pharser.get_nodes_from_documents(documents=docs)

        elif loader_type == "txt":
            docs = SimpleDirectoryReader(input_dir=dir).load_data()
            emb = get_embedding_model()
            # splitter = SemanticSplitterNodeParser(
            #     buffer_size=5, breakpoint_percentile_threshold=95, embed_model=emb
            # )

            # splitter = SentenceSplitter()
            splitter = SentenceWindowNodeParser.from_defaults(
                # how many sentences on either side to capture
                window_size=3,
                # the metadata key that holds the window of surrounding sentences
                window_metadata_key="window",
                # the metadata key that holds the original sentence
                original_text_metadata_key="original_sentence",
            )

            nodes = splitter.get_nodes_from_documents(
                documents=docs, show_progress=True
            )
        elif loader_type == "sentance":
            docs = SimpleDirectoryReader(input_dir=dir).load_data()
            splitter = SentenceSplitter(
                chunk_size=128, chunk_overlap=16, paragraph_separator="\n\n"
            )
            nodes = splitter.get_nodes_from_documents(
                documents=docs, show_progress=True
            )
        else:
            docs = []
            nodes = []

        VectorStoreIndex(nodes, storage_context=context)
        return "Index Created using documents"


@app.route("/frmIndex", methods=["POST"])
def fromIndex():
    query_text = request.json["name"]
    index_name = request.json["indexName"]

    print("Query Text -->", query_text)
    print("Index -->", index_name)

    context, db = get_elastic_storage_context(index=index_name)
    index = VectorStoreIndex(nodes=[], storage_context=context)

    retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
    # Directly retrieve results from the index
    retrieved_nodes = retriever.retrieve(query_text)

    # Process the response into a format to return
    response_data = [
        {"content": node.get_content(), "score": node.get_score()}
        for node in retrieved_nodes
    ]

    return jsonify({"response": response_data})


@app.route("/queryBasic", methods=["POST"])
def queryBasic():
    query_text = request.json["name"]
    index_name = request.json["indexName"]

    context, db = get_elastic_storage_context(index=index_name)
    index = VectorStoreIndex(nodes=[], storage_context=context)

    print("Query Text -->", query_text)
    print("Index -->", index_name)
    print("\n>>> Answer:")

    if not request.environ.get("wsgi.input").closed:
        print("Connection Closed...")

    query_engine = index.as_chat_engine(chat_mode=ChatMode.CONTEXT, verbose=True)
    response = query_engine.stream_chat(message=query_text)
    pprint_response(response)

    def generate():
        try:
            for chunk in response.response_gen:
                if chunk is None:
                    print("Received None as a chunk, skipping.")
                    continue

                if not request.environ.get("wsgi.input").closed:
                    json_data = get_formatted_response(chunk)
                    print(json_data)
                    yield f"data: {json_data}\n\n"
                else:
                    print("Client disconnected, stopping LLM call.")
                    break
        except Exception as e:
            print(f"Exception in generator: {e}")
        finally:
            print("Generator exit")

    return stream_with_context(generate())


@app.route("/query_chat", methods=["POST"])  # works really well
def query_chat():
    query_text = request.json["name"]
    index_name = request.json["indexName"]

    context, db = get_elastic_storage_context(index=index_name)
    index = VectorStoreIndex(nodes=[], storage_context=context)

    print("Query Text -->", query_text)
    print("Index -->", index_name)
    print("\n>>> Answer:")

    if not request.environ.get("wsgi.input").closed:
        print("Connection Closed...")

    retriever = index.as_retriever(retriever_mode="default", similarity_top_k=2)
    chat_engine = ContextChatEngine.from_defaults(retriever=retriever)

    # chat_engine = index.as_chat_engine(chat_mode=ChatMode.CONTEXT, verbose=True)
    response = chat_engine.stream_chat(message=query_text)
    pprint_response(response)

    def generate():
        try:
            for chunk in response.response_gen:
                if chunk is None:
                    print("Received None as a chunk, skipping.")
                    continue

                if not request.environ.get("wsgi.input").closed:
                    json_data = get_formatted_response(chunk)
                    print(json_data)
                    yield f"data: {json_data}\n\n"
                else:
                    print("Client disconnected, stopping LLM call.")
                    break
        except Exception as e:
            print(f"Exception in generator: {e}")
        finally:
            print("Generator exit")

    return stream_with_context(generate())


@app.route("/query", methods=["POST"])
def query():
    query_text = request.json["name"]
    index_name = request.json["indexName"]

    context, db = get_elastic_storage_context(index=index_name)
    index = VectorStoreIndex(nodes=[], storage_context=context)

    print("Query Text -->", query_text)
    print("Index -->", index_name)
    print("\n>>> Answer:")

    if not request.environ.get("wsgi.input").closed:
        print("Connection Closed...")

    qa_prompt_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "keep the answer short and concise\n"
        "do not include based on the provided context in the output"
    )

    text_qa_template = PromptTemplate(qa_prompt_str)

    retriever = index.as_retriever(retriever_mode="default", similarity_top_k=2)
    chat_engine = ContextChatEngine.from_defaults(
        retriever=retriever, context_template=text_qa_template
    )

    # chat_engine = index.as_chat_engine(chat_mode=ChatMode.CONTEXT, verbose=True)
    response = chat_engine.stream_chat(message=query_text)
    pprint_response(response)

    def generate():
        try:
            for chunk in response.response_gen:
                if chunk is None:
                    print("Received None as a chunk, skipping.")
                    continue

                if not request.environ.get("wsgi.input").closed:
                    json_data = get_formatted_response(chunk)
                    print(json_data)
                    yield f"data: {json_data}\n\n"
                else:
                    print("Client disconnected, stopping LLM call.")
                    break
        except Exception as e:
            print(f"Exception in generator: {e}")
        finally:
            print("Generator exit")

    return stream_with_context(generate())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5500)
