import asyncio
import os
import re
from collections import deque
from typing import List

import discord
from discord import Message
from discord.ext import commands
from dotenv import load_dotenv
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage

from browse import summarize_link
from chat import generate_response, update_response

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
RESPONSE_DELAY = int(os.getenv("RESPONSE_DELAY"))
CHAT_MAX_LEN = int(os.getenv("CHAT_MAX_LEN"))

intents = discord.Intents.all()
bot = commands.Bot(command_prefix="/", intents=intents)

message_deques = {}
history_deques = {}
processing_scheduled = {}
backfilled_channels = set()


async def process_new_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    processed_messages = []

    for message in messages:
        processed_messages.append(message)

        # Check for links in the message and generate a summarized system message if necessary
        links = re.findall(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            message.content,
        )
        if links:
            summarized_texts = await asyncio.gather(
                *[summarize_link(link) for link in links]
            )
            summarized_message_content = (
                "Here's a summary of the link(s) in the messages:\n"
                + "\n\n".join(
                    [
                        f"{link}\n{summary}"
                        for link, summary in zip(links, summarized_texts)
                    ]
                )
            )
            system_message = SystemMessage(content=summarized_message_content)
            processed_messages.append(system_message)

    return processed_messages


async def process_chat(chat_channel):
    async def respond(should_respond: bool, response: List[AIMessage]):
        if should_respond and response:
            for ai_message in response:
                await chat_channel.send(ai_message.content)
                history_deques[chat_channel.id].append(ai_message)

    global processing_scheduled

    await asyncio.sleep(RESPONSE_DELAY)

    messages = []
    while message_deques[chat_channel.id]:
        messages.append(message_deques[chat_channel.id].popleft())
    processed_messages = await process_new_messages(messages)

    should_respond, _, response = await generate_response(
        list(history_deques[chat_channel.id]), processed_messages
    )
    history_deques[chat_channel.id].extend(processed_messages)

    if not message_deques[chat_channel.id]:
        await respond(should_respond, response)
        processing_scheduled[chat_channel.id] = False
        return

    updated_should_respond = True
    updated_response = response
    prev_response = None

    while message_deques[chat_channel.id]:
        new_messages = []
        while message_deques[chat_channel.id]:
            new_messages.append(message_deques[chat_channel.id].popleft())
        processed_new_messages = await process_new_messages(new_messages)

        prev_response = updated_response
        (
            updated_should_respond,
            updated_should_send,
            updated_response,
        ) = await update_response(
            list(history_deques[chat_channel.id]),
            messages,
            updated_response,
            processed_new_messages,
        )
        messages.extend(processed_new_messages)
        history_deques[chat_channel.id].extend(processed_new_messages)

        await respond(updated_should_send, prev_response)

    await respond(updated_should_respond, updated_response)

    processing_scheduled[chat_channel.id] = False


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


async def backfill_chat(channel, limit=CHAT_MAX_LEN):
    backfilled_messages = []

    async for message in channel.history(limit=limit + 1):
        message_ = await discord_to_langchain_message(message, bot)
        backfilled_messages.append(message_)

    history_deques[channel.id].extend(
        backfilled_messages[::-1][:-1]
    )  # Reverse the order to maintain chronological order


@bot.event
async def on_ready():
    print(f"{bot.user} has connected to Discord!")


@bot.event
async def on_message(message: Message):
    global processing_scheduled

    if message.author == bot.user:
        return

    if message.channel.id not in message_deques:
        message_deques[message.channel.id] = deque()
        history_deques[message.channel.id] = deque(maxlen=CHAT_MAX_LEN)
        processing_scheduled[message.channel.id] = False

    message_ = await discord_to_langchain_message(message, bot)
    message_deques[message.channel.id].append(message_)

    if not processing_scheduled[message.channel.id]:
        processing_scheduled[message.channel.id] = True

        # Backfill channel messages
        if (
            not history_deques[message.channel.id]
            and message.channel.id not in backfilled_channels
        ):
            await backfill_chat(message.channel)
            backfilled_channels.add(message.channel.id)

        await process_chat(message.channel)

    await bot.process_commands(message)


if __name__ == "__main__":
    bot.run(TOKEN)
