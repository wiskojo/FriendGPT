import re
from typing import List, Tuple

import json5
from langchain import LLMChain
from langchain.chat_models.openai import ChatOpenAI
from langchain.schema import AIMessage, BaseMessage

from prompt import RESPONSE_PROMPT, UPDATE_PROMPT

FRIEND_NICKNAME = "GPT"


def get_chatgpt_chain(prompt, model_name="gpt-4", temperature=0, verbose=True):
    chain = LLMChain(
        llm=ChatOpenAI(model_name=model_name, temperature=temperature),
        prompt=prompt,
        verbose=verbose,
    )
    return chain


def parse_output(output: str) -> Tuple[bool, List[str]]:
    should_act = False
    messages = []

    # Extract JSON strings enclosed by triple backticks or not enclosed
    json_strings = re.findall(r"(```json)?\s*(\{.*?\})\s*(```)?", output, re.DOTALL)

    for json_str_tuple in json_strings:
        # Load the JSON string as a Python dictionary
        json_dict = json5.loads(json_str_tuple[1].strip())

        # Check if the dictionary has the "should_act" key and update should_act accordingly
        if "should_act" in json_dict:
            should_act = json_dict["should_act"]

        # Check if the dictionary has the "messages" key and update messages accordingly
        if "messages" in json_dict:
            messages.extend(json_dict["messages"])

    # Convert to AIMessage
    messages = [AIMessage(content=message) for message in messages]

    return should_act, messages


async def generate_response(
    history: List[BaseMessage], messages: List[BaseMessage]
) -> Tuple[bool, List[AIMessage]]:
    response_chain = get_chatgpt_chain(RESPONSE_PROMPT)
    output = await response_chain.apredict(
        nickname=FRIEND_NICKNAME, history=history, messages=messages
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
        nickname=FRIEND_NICKNAME,
        history=history,
        messages=messages,
        response=response,
        new_messages=new_messages,
    )
    print(output)
    return parse_output(output)
