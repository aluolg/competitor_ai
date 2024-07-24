import langchain as lc
import base64
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from email.mime import image
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain.chains.transform import TransformChain
from langchain_core.runnables import chain
from langchain_core.pydantic_v1 import BaseModel, Field
import streamlit as st

import pandas as pd
import re  

l = pd.read_csv('./scrape_list/scrape_list_formatted.csv')
l = set(l['Operator'])

##check if there are folders in the given path, if so, get all the images path in those folders

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
        azure_deployment='gpt_4o',
        api_key = st.secrets['AZURE_OPENAI_API_KEY'],
        azure_endpoint=st.secrets['AZURE_OPENAI_ENDPOINT'],
        temperature=0,
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

def get_offer(company_name: str):
    parser = StrOutputParser()
    # get folder path
    image_path = f"{company_name}/"
    
    vision_prompt = """
    Create a list of the brands that are offering promotions in the images you see. 
    List each and every promotion from these brands with headline and a few bullet 
    points summary about the offer in a table and in the table separately highlight 
    the key benefits in the next column. Add columns of information that categorize 
    them into Sports, Casino, Poker, Other product groups. Also, add a column to summarize 
    if the promotional content is targeted for customer acquisition or retention.
    """
    vision_chain = load_image_chain| image_model | parser

    return vision_chain.invoke({"image_path":f"{image_path}","prompt": vision_prompt})


# convert response to pandas dataframe and save as csv file
output = pd.DataFrame()
for n in l:
    print(f"working on {n}...")
    response = get_offer(n)

    # Use regex to extract the table part
    table_regex = re.compile(r"\|.*?\|\n(\|.*?\|\n)+")
    match = table_regex.search(response)

    if match:
        table_text = match.group(0)

        # Split the table text into lines
        lines = table_text.strip().split('\n')

        # Extract column names
        columns = [col.strip() for col in lines[0].split('|') if col.strip()]

        # Extract data
        data = []
        for line in lines[2:]:
            row = [col.strip() for col in line.split('|') if col.strip()]
            data.append(row)

        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)
        output = pd.concat([output,df])
        
    else:
        print("Table not found in the text.")
    
    print(f"done with {n}")

output.to_csv('response.csv',index=False)