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

ollama_base_url = os.getenv("OLLAMA_BASE_URL")
llm_name = os.getenv("LLM")
# Remapping for Langchain Neo4j integration

logger = get_logger(__name__)


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
    Format your output as a list of json. Each element of the list contains a pair of terms node1 and node2
    and the relation between them as edge, like the following: 
    ```[[
      
           "node_1": "A concept from extracted ontology, one or two words(representing single entity)",
           "node_2": "A related concept from extracted ontology, one or two words(representing single entity)",
           "edge": "relationship between the two concepts, node_1 and node_2, one or two words"
       ], [...]"
    ]```\
    output should be a json of these pairings no explanation required. Try to extract as many relationships as possible.
    """
    general_user_template = "context: ```{input}``` output:\n"
    messages = [
        SystemMessagePromptTemplate.from_template(general_system_template),
        HumanMessagePromptTemplate.from_template(general_user_template),
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    return qa_prompt



def main():
    
        text = """
       In the Summer of 1875, especially in the midwest, it was virtually impossible to never have heard the name: Angel Ramos aka ‘El Toro’. The Mexican immigrant robbed as many as 10 banks in a year. He wanted 'Dead Or Alive’, but the government had never been able to get him, no matter where he went. For now, however, he was lying low on a ranch over in the Northern part of Texas. That man was spectacular, it was rumored that he once shot an apple off some son of a bitch’s head from 30 yards away… with his revolver. It seemed like Angel had been blessed with luck. However, that luck finally came to an end when U.S. marshall Buck Carson finally bested him.

Carson was a different animal in his own right, his use of a gun was second to none, even Angel. He had brought down many notorious outlaws over the years, his name was revered throughout the frontier. It had almost been an honor from Uncle Sam himself to have that man on your tail.

Carson had been pursuing Angel for some time before finally tracking him to a ranch, near the abandoned town of Cedar Hollow.

At last Angel's day had come, he was working in the fields when Carson surprised him, Angel reacted just in time to avoid a lethal hit, but the bullet got him right in his abdomen. He managed to get away from Carson, before speeding off on one of the horses from a nearby stable. Angel dashed into town, mortar craters, gunshot holes, debris, and cracked buildings made up the street. Carson was right on his trail but was a bit behind him. When he turned the corner to go down Cedar Hollow's main road, Carson saw Angel's horse outside of a saloon, unhitched, and dirt shoe prints on the porch, going in.

Carson got down from his horse, drew his revolver, and cautiously walked up the porch. He pinned his back to the wall, just outside the doors and peeked into the saloon. He was very surprised to see Angel, sitting at a table, clutching his side while calmly looking out a window to his right, up into the sky. Angel's holster, with his notorious gold-plated guns: ‘Wrath & Fury’ were on the floor to his left near the main entrance. Carson, aiming down the sight of his revolver, walked through the doors, into the saloon.

Carson bent down to pick up the belt, whilst his eyes and gun were still trained on Angel. He moved on to the table that Angel was sitting at, and sat down with him. His revolver never left the sight of Angel.

“I can at least respect the fact you've made this easier for both of us,” Carson said.

“No problem. I had a pretty good run.” Angel replied

“Yes, yes you did” Carson responded.

With vigor, Angel says, “They called me El Toro”. Angel smirks and looks at Carson. “Do you want to know why they call me that?”

Not really that intrigued, Carson replies, “I guess I've got some time now”.

Continuing, Angel says, “El Toro means The Bull, and bulls are strong and ruthless by nature”

“And look at where we are now,” Carson says back.

Looking back into the window, “Yes, look at where we are now” Angel said.

Out of curiosity, Carson asks, “Now that we're here, was it… worth it? I mean why, for all of this, just to be here. I knew you must've known that this day was coming”

“You make it sound as if it is a choice.” Looking back at Carson. “Guys like me, this is how we survive, out on the run, taking down anything that gets in our way. It's all we know… no one is coming to save us” Angel said.

“It's a funny world we live in I guess” Carson replies.

“Yes,” Angel said, before going on, “Humans… were animals at heart, you cannot blame us for our nature. If you saw a Hyena eating a wildebeest, you would not imprison the Hyena, would you?”.

“We're not just animals anymore… were smart animals, we have… society, customs, order and we can't lose it, unfortunately, some have to bite the bullet to show that to others”

Smirking, “you said it,” Angle said, before continuing. “Do you believe in a heaven and hell, Mr Carson?”

“Not particularly” Carson replied.

“They say guys like me are going to hell… but it all becomes clearer to me at this moment. This was my punishment… And you, what about you? Where would you be going if the idea of a heaven or hell were true?” Angel asked.

“Well, I haven't given it much thought if I'm being honest. But if I had to guess, It'd be heaven”.

“And why is that, Mr Carson?” Angel replied.

“I fight for the greater good. That has to mean something to the man above, if he's there” Carson said.

Angel snaps back, “I tell myself the same thing, Mr Carson. We're two sides of the same coin, the only difference being we justify the bad things that we do differently, but in the end, the fact of the matter is… we're both killers, on a quest for survival.”

“Maybe you're right Mr Ramos,” Carson says, Getting up from the table, “but I'm afraid our time is up now”

Carson, standing over Angel, who looking up at him, aims the revolver at his head.

Smirking. “In the next life Mr. Carson,” Angel said.

With one pull of the trigger and a loud bang, Angel Ramos aka “El Toro” was dead.

The End.
"""
        #logger.info(text)
        prompt = get_kg_prompt()
        #prompt = ChatPromptTemplate.from_template("tell me charcter from text below: {input}")
        logger.info(prompt)
        chain =   {"input":RunnablePassthrough()}| prompt |llm | StrOutputParser()
        output = chain.invoke(text)
        logger.info(output)


if __name__ == "__main__":
    main()