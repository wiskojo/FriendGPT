import os
import re
from typing import List, Tuple

import json5
from dotenv import load_dotenv
from langchain import LLMChain
from langchain.chat_models.openai import ChatOpenAI
from langchain.schema import AIMessage, BaseMessage

from prompt import RESPONSE_PROMPT, UPDATE_PROMPT

load_dotenv()

FRIEND_NAME = os.getenv("FRIEND_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")


def get_chatgpt_chain(prompt, model_name=MODEL_NAME, temperature=0, verbose=True):
    chain = LLMChain(
        llm=ChatOpenAI(model_name=model_name, temperature=temperature),
        prompt=prompt,
        verbose=verbose,
    )
    return chain


def parse_output(output: str) -> Tuple[bool, bool, List[str]]:
    should_act = False
    should_send = False
    messages = []

    json_strings = re.findall(r"(```json)?\s*(\{.*?\})\s*(```)?", output, re.DOTALL)

    for json_str_tuple in json_strings:
        json_dict = json5.loads(json_str_tuple[1].strip())
        if "should_act" in json_dict:
            should_act = json_dict["should_act"]
        if "should_send" in json_dict:
            should_send = json_dict["should_send"]
        if "messages" in json_dict:
            messages.extend(json_dict["messages"])

    messages = [AIMessage(content=message) for message in messages]

    return should_act, should_send, messages


async def generate_response(
    history: List[BaseMessage], messages: List[BaseMessage]
) -> Tuple[bool, List[AIMessage]]:
    response_chain = get_chatgpt_chain(RESPONSE_PROMPT)
    output = await response_chain.apredict(
        name=FRIEND_NAME, history=history, messages=messages
    )
    print(output)
    return parse_output(output)


async def update_response(
    history: List[BaseMessage],
    messages: List[BaseMessage],
    response: List[AIMessage],
    new_messages: List[BaseMessage],
) -> Tuple[bool, List[AIMessage]]:
    update_chain = get_chatgpt_chain(UPDATE_PROMPT)
    output = await update_chain.apredict(
        name=FRIEND_NAME,
        history=history,
        messages=messages,
        response=response,
        new_messages=new_messages,
    )
    print(output)
    return parse_output(output)
