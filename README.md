# Retrieval-Augmented Generation (RAG) POC

This web application demonstrates a Proof of Concept (POC) for Retrieval-Augmented Generation (RAG).

## Endpoints

The application exposes the following endpoints:

1. **`/load`** - Processes documents and creates embeddings.
2. **`/query`** - HTTP streaming API to support chat with an LLM.

## Tech Stack

- Python
- Llamaindex
- Flask
- ELK Stack


## Project Setup

- Dependencies are manage using poetry
- install poetry
```pipx install poetry```
- Run ```poetry install --sync``` to install all dependencies
- Run ```poetry shell``` to activate virtual environment
- Run ```flask run``` to start the web server



## Misc

- Add new dependency ```poetry add <package>```
- Show all dependencies ```poetry show``` or ```poetry tree```


