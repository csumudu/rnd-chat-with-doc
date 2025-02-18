from llama_index.core import VectorStoreIndex
from IPython.display import Markdown, display

from rnd_chat_with_doc.modules.manage.document_uploader import upload_documents
from rnd_chat_with_doc.modules.manage.vector_storage_context_manager import (
    get_chroma_storage_context,
    get_elastic_storage_context,
)


def chat():
    chroma_storage_context = get_chroma_storage_context()
    elastic_storage_context = get_elastic_storage_context()

    nodes = upload_documents()
    index = VectorStoreIndex(
        nodes=nodes,
        show_progress=True,
        use_async=True,
        storage_context=elastic_storage_context,  # chroma_storage_context,
    )
    qe = index.as_query_engine()

    while True:
        q = input("Question :")
        if q != "quit":
            res = qe.query(q)
            display(res)


if __name__ == "__main__":
    chat()
