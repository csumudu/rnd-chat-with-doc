from llama_index.core import (
    download_loader,
    SimpleDirectoryReader,
    Settings,
    VectorStoreIndex,
)
from llama_index.llms.openai import OpenAI
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModeModel

from llama_index.readers.file.unstructured import UnstructuredReader
from llama_index.readers.web import SimpleWebPageReader

from rnd_chat_with_doc.modules.manage.vector_storage_context_manager import (
    get_elastic_storage_context,
)
from rnd_chat_with_doc.modules.util.file_utils import get_abs_path
import os
import nltk

from rnd_chat_with_doc.modules.util.llama_index_utils import (
    set_llamaindex_logging,
    set_llm,
)


nltk.download("averaged_perceptron_tagger_eng")


def main():
    # path = get_abs_path(__file__, "../docs/csgi")
    # print("src->", path)

    # docs = SimpleDirectoryReader(
    #     # input_files=[get_abs_path(__file__, "../docs/csgi/ConfigGuide.html")],
    #     input_files=[get_abs_path(__file__, "../docs/csgi/UserGuide.html")],
    #     file_extractor={".html": UnstructuredReader()},
    # ).load_data()

    path = get_abs_path(__file__, "../out")
    print("dir path-->",path)
    
    docs = SimpleDirectoryReader(input_dir=path).load_data()


    # for doc in docs:
    #     src_file_path = doc.metadata.get(
    #         "file_path"
    #     )  # Adjust this based on your actual metadata structure
    #     if src_file_path:
    #         base_file_name = os.path.basename(src_file_path)
    #         output_file_name = os.path.splitext(base_file_name)[0] + ".txt"
    #         output_file_path = os.path.join(os.getcwd(), "out", output_file_name)
    #         print("out->", output_file_path)

    #         with open(
    #             output_file_path, "w", encoding="utf-8", errors="replace"
    #         ) as file:
    #             file.write(doc.text)

    pharser = SimpleNodeParser.from_defaults(
        paragraph_separator="\n\n"
    )

    Settings.node_parser = pharser

    set_llamaindex_logging()

    context, db = get_elastic_storage_context("unstructured_index")

    VectorStoreIndex.from_documents(
        documents=docs, storage_context=context, show_progress=True
    )

    print("Data Loading completed...")
    db.close()


def convert_to_txt():
    docs = SimpleDirectoryReader(
        input_dir=get_abs_path(__file__, "../docs/guides"),
        file_extractor={".html": UnstructuredReader()},
    ).load_data()

    # docs = SimpleWebPageReader(html_to_text=True).load_data(
    #     [
    #         "http://localhost:8080/ConfigGuide.html"
    #     ]
    # )

    pharser = SimpleNodeParser.from_defaults()

    Settings.node_parser = pharser

    for doc in docs:
        src_file_path = doc.metadata.get(
            "file_path"
        )  # Adjust this based on your actual metadata structure
        if src_file_path:
            base_file_name = os.path.basename(src_file_path)
            output_file_name = os.path.splitext(base_file_name)[0] + ".txt"
            output_file_path = os.path.join(os.getcwd(), "out", output_file_name)
            print("out->", output_file_path)

            with open(
                output_file_path, "w", encoding="utf-8", errors="replace"
            ) as file:
                file.write(doc.text)

    print("Conversion completed.....")


if __name__ == "__main__":
    set_llm()
    main()
    # convert_to_txt()
