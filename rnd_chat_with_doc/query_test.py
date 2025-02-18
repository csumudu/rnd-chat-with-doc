from llama_index.core import (
    VectorStoreIndex,
)

from rnd_chat_with_doc.modules.manage.vector_storage_context_manager import (
    get_elastic_storage_context,
)
from rnd_chat_with_doc.modules.util.llama_index_utils import set_llamaindex_logging
from llama_index.core.chat_engine.types import ChatMode


def get_index():
    set_llamaindex_logging()

    context, db = get_elastic_storage_context("unstructured_index")
    index = VectorStoreIndex.from_vector_store(vector_store=db, show_progress=True)

    return index, db


def get_chat_engine():
    index, db = get_index()
    ce = index.as_chat_engine(chat_mode=ChatMode.CONTEXT, verbose=True)
    return ce


def query():
    index, db = get_index()
    qe = index.as_query_engine()
    res = qe.query("What is direct variables")

    print(res)

    db.close()


if __name__ == "__main__":
    query()
