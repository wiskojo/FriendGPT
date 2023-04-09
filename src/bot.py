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
history_deque = deque(maxlen=30)
processing_scheduled = False


async def process_chat(chat_channel):
    global processing_scheduled

    await asyncio.sleep(DELAY)

    messages = list(message_deque)
    message_deque.clear()

    should_respond, response = await generate_response(list(history_deque), messages)
    if should_respond:
        updated_should_respond = True
        updated_response = response
        while updated_should_respond:
            new_messages = list(message_deque)
            message_deque.clear()
            if new_messages:
                updated_should_respond, updated_response = await update_response(
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


async def discord_to_langchain_message(
    message: Message, bot: commands.Bot
) -> BaseMessage:
    async def format_message_content(message: Message) -> str:
        author = message.author.display_name
        timestamp = message.created_at.strftime("%Y-%m-%d %H:%M:%S")

        content = f"{author} ({timestamp}): {message.content}"

        if message.reference:
            try:
                replied_message = await message.channel.fetch_message(
                    message.reference.message_id
                )
                replied_author = replied_message.author.display_name
                replied_content = replied_message.content
                content += f"\n> In reply to {replied_author}: {replied_content}"
            except:
                content += "\n> In reply to a deleted message"

        if message.attachments:
            attachments = []
            for attachment in message.attachments:
                attachments.append(f"Attachment: {attachment.url}")
            attachments_str = "\n".join(attachments)
            content += f"\n{attachments_str}"

        if message.embeds:
            embeds = []
            for embed in message.embeds:
                embeds.append(f"Embed: {embed.title or ''} - {embed.description or ''}")
            embeds_str = "\n".join(embeds)
            content += f"\n{embeds_str}"

        return content

    if message.author == bot.user:
        return AIMessage(content=message.content)
    else:
        formatted_message = await format_message_content(message)
        return HumanMessage(content=formatted_message)


@bot.event
async def on_ready():
    print(f"{bot.user} has connected to Discord!")


@bot.event
async def on_message(message: Message):
    global processing_scheduled

    if message.author == bot.user:
        return

    message_ = await discord_to_langchain_message(message, bot)
    message_deque.append(message_)

    if not processing_scheduled:
        processing_scheduled = True
        await process_chat(message.channel)

    await bot.process_commands(message)


if __name__ == "__main__":
    bot.run(TOKEN)
