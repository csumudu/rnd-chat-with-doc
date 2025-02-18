from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os

key = os.getenv("NVIDIA_KEY")
llm = ChatNVIDIA(model="mixtral_8x7b", nvidia_api_key=key)
result = llm.invoke("Write me a small song on AI")
print(result.content)