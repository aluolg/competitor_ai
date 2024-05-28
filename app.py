import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatMessagePromptTemplate
import langchain as lc
import base64
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from email.mime import image
from langchain_openai import AzureChatOpenAI
from langchain.chains.transform import TransformChain
from langchain_core.runnables import chain
from langchain_core.pydantic_v1 import BaseModel, Field


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(
            content="Hello! I am Competitor Intelligent AI. I can help you with your questions. How can I help you today?"
        )
    ]

st.set_page_config(page_title="Competitor Intelligent AI", page_icon=":shark:", layout="wide")
st.title("Competitor Intelligent AI")


def get_all_files_path(directory):
    files = []
    # check if folder called screenshot exist
    if os.path.exists(os.path.join(directory,"screenshot")):
        files.extend(get_files(os.path.join(directory,"screenshot")))

    return files

def get_files(directory):
    files = []
    for root, _, file in os.walk(directory):
        for f in file:
            files.append(os.path.join(root,f))
    return files


# encode image to base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def load_image(inputs: dict) -> dict:
    """Load image from file and encode it as base64."""
    image_path = get_all_files_path(inputs["image_path"])

    image_list = []

    for path in image_path:
        image_base64 = encode_image(path)
        image_list.append(image_base64)

    return {"image":image_list}


load_image_chain= TransformChain(
    input_variables=["image_path"],output_variables=["image"],transform=load_image
)

@chain
def image_model(inputs: dict) -> str | list[str]| dict:
    """Invoke model with image and prompt."""
    model = AzureChatOpenAI(
        openai_api_version="2023-05-15",
        azure_deployment='vision_model',
        api_key=st.secrets.api_key,
        azure_endpoint=st.secrets.azure_endpoint,
        temperature=0.1,
        max_tokens=4096,
    )
    msg = model.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": inputs["prompt"]},
                    *[
                        {
                            "type": "image_url",
                            "image_url": {"url":f"data:image/jpeg;base64,{image}"},
                        }
                        for image in inputs["image"]
                    ],
                ]
            )
        ]
    )
    return msg.content

def get_offer(company_name: str,user_query:str):
    parser = StrOutputParser()
    ##get folder path base on place_id
    image_path = f"{company_name}/"
    vision_prompt = user_query
    vision_chain = load_image_chain| image_model | parser
    return vision_chain.invoke({"image_path":f"{image_path}","prompt": vision_prompt})



for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
        
user_query=st.chat_input("Ask me anything!")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):

        response = get_offer(company_name='betmgm',user_query=user_query)
        st.markdown(response)

    st.session_state.chat_history.append(AIMessage(content=response))


