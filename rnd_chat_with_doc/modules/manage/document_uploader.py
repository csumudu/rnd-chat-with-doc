from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.extractors import SummaryExtractor
from llama_index.embeddings.openai import OpenAIEmbedding


from rnd_chat_with_doc.modules.config.global_settings import get_config
from rnd_chat_with_doc.modules.util.cache_manager import get_cache
from rnd_chat_with_doc.modules.util.file_utils import get_abs_path
from rnd_chat_with_doc.modules.util.llm_manager import get_llm
from rnd_chat_with_doc.modules.util.logging_manager import log_action


def upload_documents():
    config = get_config()

    llm = get_llm()

    documents = SimpleDirectoryReader(
        input_dir=config.get("STORAGE_PATH"), filename_as_id=True
    ).load_data()

    for d in documents:
        log_action(type="DOCUMENT", action=f"Document {d.doc_id} uploaded.")

    chached_hash = get_cache()

    pipeline = IngestionPipeline(
        transformations=[
            TokenTextSplitter(chunk_size=1024, chunk_overlap=20),
            SummaryExtractor(llm=llm, summaries=["self"]),
            OpenAIEmbedding(),
        ],
        cache=chached_hash,
    )

    nodes = pipeline.run(documents=documents)
    pipeline.cache.persist(persist_path=config.get("CACHE_FILE"))

    log_action(type="LLM", action=f"{len(nodes)} Nodes generated")
    return nodes


if __name__ == "__main__":
    upload_documents()
