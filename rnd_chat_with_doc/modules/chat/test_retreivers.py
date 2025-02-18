from llama_index.core.schema import TextNode
from llama_index.core import SummaryIndex, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SimpleFileNodeParser, SimpleNodeParser
from llama_index.core import Settings

from rnd_chat_with_doc.modules.config.global_settings import get_config
from rnd_chat_with_doc.modules.manage.vector_storage_context_manager import (
    get_elastic_storage_context,
)
from rnd_chat_with_doc.modules.util.llm_manager import (
    async_get_local_embeddings,
    get_embedding_model,
    get_local_embeddings,
    get_local_openai_embeddind,
)
from rnd_chat_with_doc.modules.util.logging_manager import log

emb_mod = get_embedding_model()

Settings.embed_model = emb_mod


async def get_local_embeddings_async(txt):
    return [0, 0]


def custom(text):
    return text


class CustomEmbeddingModel:
    def __init__(self):
        # Initialize your model or any required components here
        pass

    async def aget_text_embedding_batch(self, texts, **kwargs):
        # Your async logic to get embeddings for a batch of texts
        embeddings = [await async_get_local_embeddings(text) for text in texts]
        return embeddings

    def get_agg_embedding_from_queries(self, queries):
        # Aggregate embeddings from multiple queries
        # For simplicity, you could just sum the embeddings or average them
        embeddings = [self.get_text_embedding(query) for query in queries]
        agg_embedding = [sum(x) / len(embeddings) for x in zip(*embeddings)]
        return agg_embedding

    def get_text_embedding(self, text):
        # Synchronous method if needed
        return get_local_embeddings(text)


def test_retreiver():
    movieSummaries = [
        TextNode(
            text="Andy Dufresne, a successful banker, is sentenced to life imprisonment for the murder of his wife and her lover, despite his claims of innocence. In Shawshank Prison, he befriends Ellis 'Red' Redding and earns the respect of his fellow inmates through his quiet strength, integrity, and resilience. Over two decades, Andy endures the brutal conditions of prison life while secretly working on a plan for freedom and justice, ultimately achieving a miraculous escape that inspires hope and redemption.",
            metadata={"title": "The Shawshank Redemption"},
        ),
        TextNode(
            text="The story of the powerful Corleone crime family, led by Vito Corleone, explores the dynamics of power, loyalty, and betrayal within the world of organized crime. As Vito's health declines, his youngest son, Michael, reluctantly steps into the family's illicit operations, initially seeking a life away from crime. However, after a series of violent events, Michael is drawn deeper into the family's criminal empire, eventually becoming a ruthless leader who will do whatever it takes to protect the family legacy.",
            metadata={"title": "The Godfather"},
        ),
        TextNode(
            text="In the chaotic streets of Gotham City, Batman, a masked vigilante, battles against rising crime and corruption. His nemesis, the Joker, a sadistic criminal mastermind, unleashes a reign of terror, pushing Batman to the limits of his morality and resolve. As the Joker's plans unfold, Batman must confront his own inner demons and make agonizing choices to save the city and the people he loves. The film explores the fine line between heroism and vigilantism, and the price of maintaining order in a world on the brink of chaos.",
            metadata={"title": "The Dark Knight"},
        ),
        TextNode(
            text="A series of interconnected stories unfold in the underbelly of Los Angeles, where mobsters, small-time criminals, and enigmatic characters navigate a world of violence, redemption, and dark humor. The film weaves together the lives of a hitman with a moral crisis, his partner, their boss's wife, and a washed-up boxer, all bound by a series of twists and turns. With its nonlinear narrative and sharp dialogue, 'Pulp Fiction' challenges conventional storytelling and offers a gritty yet stylish exploration of fate and choice.",
            metadata={"title": "Pulp Fiction"},
        ),
        TextNode(
            text="Dom Cobb, a skilled thief who specializes in the art of 'extraction,' is tasked with planting an idea into the mind of a corporate heir through a process known as 'inception.' To achieve this, Cobb assembles a team of experts to delve into the complex layers of dreams, where reality and illusion blur. As they navigate through the dreamscapes, Cobb faces his own psychological battles, haunted by the memory of his deceased wife. 'Inception' explores the boundaries of consciousness, the nature of reality, and the consequences of living in a world of dreams.",
            metadata={"title": "Inception"},
        ),
    ]

    dir = get_config().get("LARGE_DOC_STORAGE")
    log(f"Directory Path: {dir}")

    docs = SimpleDirectoryReader(input_dir=dir).load_data()
    pharser = SimpleNodeParser.from_defaults()
    doc_nodes = pharser.get_nodes_from_documents(documents=docs)

    context, db = get_elastic_storage_context()
    # index = SummaryIndex.from_documents(documents=docs, storage_context=context,db=db)

    model = CustomEmbeddingModel()
    vec_index = VectorStoreIndex(
        nodes=doc_nodes,
        show_progress=True,
        use_async=True,
        embed_model=model,
        storage_context=context,  # chroma_storage_context,
    )

    ret = vec_index.as_retriever(retriever_mode="embedding")
    res = ret.retrieve("Context Retrieval")
    log("Res Len", len(res))
    print(res[0].text)

    db.close()


if __name__ == "__main__":
    test_retreiver()
