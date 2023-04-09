# FriendGPT
**FriendGPT** is an AI-powered chatbot for Discord that strives to provide a unique and engaging experience by participating in group chats more naturally. Unlike traditional Discord GPT bots that require specific commands to interact with users, FriendGPT aims to act as an autonomous and helpful friend in the group chat, intelligently figuring out when to engage in the conversation.

## Features
1. **Autonomous group chat engagement:** FriendGPT is designed to participate in group chat settings without the need for specific commands. It can handle multiple users talking at the same time and determine when it should contribute to the conversation.

2. **Fluid conversation handling:** Instead of responding to messages on a one-to-one basis, FriendGPT tries to batch messages together to provide more fluid and coherent responses. This approach aims to make the chatbot feel more like a natural participant in the group chat.

## Note

1. **High resource consumption:** FriendGPT tends to operate better on GPT-4. But due to the need to consume all messages in the server, FriendGPT can be resource-intensive. Keep this in mind when deploying and using the chatbot in your Discord server.

## Getting Started
To get started with FriendGPT, follow these steps:

1. Clone the repository and install the required dependencies.
2. Set up your `DISCORD_TOKEN` and `OPENAI_API_KEY` in the `.env` file.
3. Run the `bot.py` script to start the chatbot.
4. Invite the chatbot to your desired Discord server and start chatting!