import logging
import sys
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.vector_stores import VectorStoreInfo, MetadataInfo
from llama_index.vector_stores.elasticsearch import ElasticsearchStore

from rnd_chat_with_doc.modules.manage.vector_storage_context_manager import (
    get_elastic_storage_context,
)
from rnd_chat_with_doc.modules.util.llm_manager import get_llm

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


async def test_elastic():
    llm = get_llm()
    Settings.llm = llm

    nodes = [
        TextNode(
            text=(
                "A bunch of scientists bring back dinosaurs and mayhem breaks" " loose"
            ),
            metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
        ),
        TextNode(
            text=(
                "Leo DiCaprio gets lost in a dream within a dream within a dream"
                " within a ..."
            ),
            metadata={
                "year": 2010,
                "director": "Christopher Nolan",
                "rating": 8.2,
            },
        ),
        TextNode(
            text=(
                "A psychologist / detective gets lost in a series of dreams within"
                " dreams within dreams and Inception reused the idea"
            ),
            metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
        ),
        TextNode(
            text=(
                "A bunch of normal-sized women are supremely wholesome and some"
                " men pine after them"
            ),
            metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
        ),
        TextNode(
            text="Toys come alive and have a blast doing so",
            metadata={"year": 1995, "genre": "animated"},
        ),
    ]

    vector_store_info = VectorStoreInfo(
        content_info="Brief summary of a movie",
        metadata_info=[
            MetadataInfo(
                name="genre",
                description="The genre of the movie",
                type="string or list[string]",
            ),
            MetadataInfo(
                name="year",
                description="The year the movie was released",
                type="integer",
            ),
            MetadataInfo(
                name="director",
                description="The name of the movie director",
                type="string",
            ),
            MetadataInfo(
                name="rating",
                description="A 1-10 rating for the movie",
                type="float",
            ),
        ],
    )

    context, db = get_elastic_storage_context("cc-elk-embeddings")
    index = VectorStoreIndex(nodes, storage_context=context)

    ret = VectorIndexAutoRetriever(index=index, vector_store_info=vector_store_info)
    res =await ret.aretrieve("List all animated movies")

    print(res)

    db.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_elastic())
