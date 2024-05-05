import os

import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama

from PyPDF2 import PdfReader
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Neo4jVector
from streamlit.logger import get_logger

# load api key lib
from dotenv import load_dotenv

load_dotenv(".env")


url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM")
# Remapping for Langchain Neo4j integration
os.environ["NEO4J_URL"] = url

logger = get_logger(__name__)

logger.info(ollama_base_url)
#using OllamaEmbeddings
embeddings = OllamaEmbeddings(
            base_url=ollama_base_url, model=llm_name
        )
dimension = 4096

#for chat handling from streamlit
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

#loading the llm model running in local using Ollama
llm = ChatOllama(
            temperature=0,
            base_url=ollama_base_url,
            model=llm_name,
            streaming=True,
            # seed=2,
            top_k=10,  # A higher value (100) will give more diverse answers, while a lower value (10) will be more conservative.
            top_p=0.3,  # Higher value (0.95) will lead to more diverse text, while a lower value (0.5) will generate more focused text.
            num_ctx=3072,  # Sets the size of the context window used to generate the next token.
    )


def main():
    st.header("Upload ur file and start chatting")

    # upload a your pdf file
    pdf = st.file_uploader("Upload", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        logger.info("splitting the input text document...........")
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        #logger.info(text)

        
        # langchain_textspliter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )

        logger.info("splitting uploaded content into chunks..........")
        chunks = text_splitter.split_text(text=text)

        logger.info("saving the chunks as vectors in Neo4j......")
        # Store the chunks part in db (vector)
        vectorstore = Neo4jVector.from_texts(
            chunks,
            url=url,
            username=username,
            password=password,
            embedding=embeddings,
            index_name="pdf_bot",
            node_label="PdfBotChunk",
            pre_delete_collection=True,  # Delete existing PDF data
        )

        logger.info("using retriever from langchain......")
        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
        )

        # Accept user questions/query
        query = st.text_input("Ask questions about your file")

        if query:
            stream_handler = StreamHandler(st.empty())
            logger.info("using retriever from langchain......")
            qa.run(query, callbacks=[stream_handler])


if __name__ == "__main__":
    main()
