from llama_index.core.indices import load_index_from_storage

from rnd_chat_with_doc.modules.manage.vector_storage_context_manager import get_elastic_storage_context

def load():
    context,db = get_elastic_storage_context(index="cc-elk-embeddings")
    indx = load_index_from_storage(storage_context=context,index_id="cc-elk-embeddings")

    print(len(indx))


if __name__ =="__main__":
    load()