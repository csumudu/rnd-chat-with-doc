from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

from rnd_chat_with_doc.modules.util.llm_manager import get_embedding_model, get_llm, get_nim_embedding_model, get_nim_llm


def set_llamaindex_logging():
    debug = LlamaDebugHandler(print_trace_on_end=True)
    cbManager = CallbackManager(handlers=[debug])

    Settings.callback_manager = cbManager


def set_llm():
    llm = get_llm()
    embed_model = get_embedding_model()
    Settings.llm = llm
    Settings.embed_model = embed_model


def set_llm_nvidia():
    llm = get_nim_llm()
    emb = get_nim_embedding_model()
    Settings.llm = llm
    Settings.embed_model = emb

