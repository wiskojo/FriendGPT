import asyncio
import os
from collections import deque

import discord
from discord import Message
from discord.ext import commands
from dotenv import load_dotenv
from langchain.schema import AIMessage, BaseMessage, HumanMessage

from chat import generate_response, update_response

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.all()
bot = commands.Bot(command_prefix="/", intents=intents)

DELAY = 10
message_deque = deque()
history_deque = deque()
processing_scheduled = False


async def process_chat(chat_channel):
    global processing_scheduled

    await asyncio.sleep(DELAY)

    messages = list(message_deque)
    message_deque.clear()

    should_respond, response = generate_response(list(history_deque), messages)
    if should_respond:
        updated_should_respond = True
        updated_response = response
        while updated_should_respond:
            new_messages = list(message_deque)
            message_deque.clear()
            if new_messages:
                updated_should_respond, updated_response = update_response(
                    list(history_deque), messages, updated_response, new_messages
                )
                messages.extend(new_messages)
            else:
                break

        if updated_response:
            for ai_message in updated_response:
                await chat_channel.send(ai_message.content)
            history_deque.extend(messages)
            history_deque.extend(updated_response)

    processing_scheduled = False


def discord_to_langchain_message(message: Message, bot: commands.Bot) -> BaseMessage:
    def format_message_content(message: Message) -> str:
        return message.content

    if message.author == bot.user:
        return AIMessage(content=format_message_content(message))
    else:
        return HumanMessage(content=format_message_content(message))


@bot.event
async def on_ready():
    print(f"{bot.user} has connected to Discord!")


@bot.event
async def on_message(message: Message):
    global processing_scheduled

    message_deque.append(discord_to_langchain_message(message, bot))

    if message.author == bot.user:
        return

    if not processing_scheduled:
        processing_scheduled = True
        await process_chat(message.channel)

    await bot.process_commands(message)


if __name__ == "__main__":
    bot.run(TOKEN)
