from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex


from rnd_chat_with_doc.modules.manage.vector_storage_context_manager import (
    get_chroma_collection,
    get_elastic_storage_context,
)
from rnd_chat_with_doc.modules.util.llm_manager import (
    async_get_embedding_model,
    async_get_llm,
    get_embedding_model,
    get_llm,
)


def embed():
    emb = OpenAIEmbedding()
    a = emb.get_text_embedding("Monkey")
    b = emb.get_text_embedding("Human")

    # dot_product cosine euclidean
    c = emb.similarity(a, b, mode="euclidean")
    print(c)


def test_chroma_index():
    col = get_chroma_collection()
    res = col.get()

    emb = OpenAIEmbedding()
    embedding = emb.get_text_embedding("NLP")

    qRes = col.query(
        query_embeddings=embedding, n_results=1, include=["distances", "embeddings"]
    )

    print(qRes)


async def test_elastic():
    movies = [
        TextNode(
            text="The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.",
            metadata={"title": "Pulp Fiction"},
        ),
        TextNode(
            text="When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",
            metadata={"title": "The Dark Knight"},
        ),
        TextNode(
            text="An insomniac office worker and a devil-may-care soapmaker form an underground fight club that evolves into something much, much more.",
            metadata={"title": "Fight Club"},
        ),
        TextNode(
            text="A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into thed of a C.E.O.",
            metadata={"title": "Inception"},
        ),
        TextNode(
            text="A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.",
            metadata={"title": "The Matrix"},
        ),
        TextNode(
            text="Two detectives, a rookie and a veteran, hunt a serial killer who uses the seven deadly sins as his motives.",
            metadata={"title": "Se7en"},
        ),
        TextNode(
            text="An organized crime dynasty's aging patriarch transfers control of his clandestine empire to his reluctant son.",
            metadata={"title": "The Godfather", "theme": "Mafia"},
        ),
    ]
    storage_context,db = get_elastic_storage_context("test-embeddings")

    llm = await async_get_llm()
    embed_model = await async_get_embedding_model()

    index = VectorStoreIndex(
        nodes=movies, storage_context=storage_context, embed_model=embed_model
    )

    qe = index.as_query_engine(llm)
    res = await qe.aquery("list all movie titles")

    print(res)
    db.close()


    


if __name__ == "__main__":
    # embed()
    import asyncio
    asyncio.run(test_elastic())


