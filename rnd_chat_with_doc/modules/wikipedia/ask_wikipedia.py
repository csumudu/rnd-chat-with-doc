from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import SummaryIndex
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding

from rnd_chat_with_doc.modules.util.llm_manager import get_llm

import logging

logging.basicConfig(level=logging.DEBUG)

def query_wikipedia(topic: str) -> str:
    llm = get_llm()
    Settings.llm = llm

    loader = WikipediaReader()
    load_kwargs = {
        "auto_suggest": True,  # Example of a keyword argument
        "redirect": True,  # Another example
    }
    documents = loader.load_data(pages=["Sri Lanka"], **load_kwargs)


    pharser = SimpleNodeParser.from_defaults()
    nodes = pharser.get_nodes_from_documents(documents)

    qe = SummaryIndex(nodes[0:5]).as_query_engine()

    res = qe.query(topic)

    return res


if __name__ == "__main__":
    while True:
        topic = input("Enter Topic : ")

        if topic == "exit":
            break

        print("Answer: ", query_wikipedia(topic=topic))

    print("hello")
