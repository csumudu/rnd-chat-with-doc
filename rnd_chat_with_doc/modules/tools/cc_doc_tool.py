from rnd_chat_with_doc.query_test import get_index


def customer_connect_documentation(query: str) -> str:
    """
    Answer queries using customer connect documentation
    """
    index, db = get_index()
    qe = index.as_query_engine()
    res = qe.query(query)

    return res


def character_counter(text:str)->int:
    """
    Count the number of characters in the given text
    """
    return len(text)
