from llama_index.core.schema import TextNode
from llama_index.core import SimpleDirectoryReader
from llama_index.core.extractors import SummaryExtractor, QuestionsAnsweredExtractor
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.schema import TransformComponent
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings

from rnd_chat_with_doc.modules.util.file_utils import get_abs_path
from rnd_chat_with_doc.modules.util.llm_manager import get_llm


class CustomTransformation(TransformComponent):

    def __call__(self, nodes, **kwargs):
        print("called-----")

        meta_adapter = lambda n, updates: TextNode(
            **{**n.__dict__, "metadata": {**n.metadata, **updates}}
        )

        nodes = [meta_adapter(n, {"author": "Sumudu", "type": "Book"}) for n in nodes]

        nodes = [
            (
                lambda n: TextNode(
                    **{**n.__dict__, "metadata": {**n.metadata, "Degree": "MIS"}}
                )
            )(n)
            for n in nodes
        ]

        return nodes


def transform():
    llm = get_llm()
    Settings.llm = llm

    docFolderPath = get_abs_path(__file__, "docs/tiny", 3)
    cachePath = get_abs_path(__file__, "store/ingestion_cache.json", 3)

    reader = SimpleDirectoryReader(docFolderPath)

    documents = reader.load_data()

    try:
        cached_hashes = IngestionCache.from_persist_path(cachePath)
        print("Cache file found. Running using cache...")
    except:
        cached_hashes = ""
        print("No cache file found. Running without cache...")

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(),
            TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=128),
            SummaryExtractor(),
            QuestionsAnsweredExtractor(questions=1),
            CustomTransformation(),
        ],
        cache=cached_hashes,
    )

    nodes = pipeline.run(documents=documents, show_progress=True)
    pipeline.cache.persist(cachePath)

    print("All documents loaded")

    for n in nodes[0:5]:
        print("meta-->", n.metadata)


if __name__ == "__main__":
    transform()
