import asyncio
import os
from collections import deque

import discord
from discord.ext import commands
from dotenv import load_dotenv

from chat import generate_response, update_response

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.all()
bot = commands.Bot(command_prefix="!", intents=intents)

DELAY = 5
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
                    list(history_deque), messages, new_messages
                )
                messages.extend(new_messages)
            else:
                break

        if updated_response:
            await chat_channel.send(updated_response)
            history_deque.extend(messages)
            history_deque.append(updated_response)

    processing_scheduled = False


@bot.event
async def on_ready():
    print(f"{bot.user} has connected to Discord!")


@bot.event
async def on_message(message):
    global processing_scheduled

    if message.attachments:
        content = f"{message.content}\n{message.attachments[0].url}"
    else:
        content = message.content

    message_deque.append(content)

    if message.author == bot.user:
        return

    if not processing_scheduled:
        processing_scheduled = True
        await process_chat(message.channel)

    await bot.process_commands(message)


if __name__ == "__main__":
    bot.run(TOKEN)
