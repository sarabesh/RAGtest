import os
import streamlit as st
from streamlit.logger import get_logger
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from PyPDF2 import PdfReader
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)

# load api key lib
from dotenv import load_dotenv

load_dotenv("local.env")


url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM")
# Remapping for Langchain Neo4j integration
os.environ["NEO4J_URL"] = url

print(username)
print(password)
logger = get_logger(__name__)
graphs = Neo4jGraph(url=url,username=username,password=password)


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

# llm_transformer = LLMGraphTransformer(llm=llm)


def get_kg_prompt():
    # RAG response
    #   System: Always talk in pirate speech.
    general_system_template = """ 
    You are a network graph maker who extracts terms and their relations from a given context. 
    You are provided with a context chunk (delimited by ```) Your task is to extract the ontology 
    of terms mentioned in the given context. These terms should represent the key concepts as per the context.
    Thought 1: While traversing through each sentence, Think about the key terms mentioned in it.
        Terms may include object, entity, location, organization, person,
        condition, acronym, documents, service, concept, etc.
        Terms should be as atomistic as possible
    Thought 2: Think about how these terms can have one on one relation with other terms.
        Terms that are mentioned in the same sentence or the same paragraph are typically related to each other.
        Terms can be related to many other terms
    Thought 3: Find out the relation between each such related pair of terms. 
    Format your output as a list of json. Each element of the list contains a pair of terms
    and the relation between them, like the follwing: 
    ```[[
      
           "node_1": "A concept from extracted ontology",
           "node_2": "A related concept from extracted ontology",
           "edge": "relationship between the two concepts, node_1 and node_2 in one or two sentences"
       ], [...]"
    ]```
    """
    general_user_template = "context: ```{input}```"
    messages = [
        SystemMessagePromptTemplate.from_template(general_system_template),
        HumanMessagePromptTemplate.from_template(general_user_template),
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    return qa_prompt



def main():
    
    st.header("Upload ur file and check neo4j")

    # upload a your pdf file
    pdf = st.file_uploader("Upload", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        logger.info("splitting the input text document...........")
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        #logger.info(text)
        prompt = get_kg_prompt()
        logger.info(prompt)
        chain =  prompt | llm | StrOutputParser()
        chain.invoke({"input":text})
        #logger.info(llm_output)
        # graph_documents = llm_transformer.convert_to_graph_documents(documents)
        # print(f"Nodes:{graph_documents[0].nodes}")
        # print(f"Relationships:{graph_documents[0].relationships}")

        # graphs_extracted = 
        # graphs.add_graph_documents(graphs_extracted)



if __name__ == "__main__":
    main()