import asyncio

from bs4 import BeautifulSoup
from langchain import PromptTemplate
from playwright.async_api import async_playwright

from chat import get_chatgpt_chain


async def scrape_text(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch()

        # Set up a more complete user agent and viewport size
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36",
            viewport={"width": 1280, "height": 800},
        )

        page = await context.new_page()

        try:
            await page.goto(url)

            # Add delay or wait for specific elements if necessary
            await asyncio.sleep(2)

            content = await page.content()
            soup = BeautifulSoup(content, "html.parser")

            for script in soup(["script", "style"]):
                script.extract()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)

            return text
        finally:
            await browser.close()


def split_text(text, max_length=3000):
    paragraphs = text.split("\n")
    current_length = 0
    current_chunk = []

    for paragraph in paragraphs:
        if current_length + len(paragraph) + 1 <= max_length:
            current_chunk.append(paragraph)
            current_length += len(paragraph) + 1
        else:
            yield "\n".join(current_chunk)
            current_chunk = [paragraph]
            current_length = len(paragraph) + 1

    if current_chunk:
        yield "\n".join(current_chunk)


async def summarize_text(text, combine_summary=False, verbose=False):
    if not text:
        return "Error: No text to summarize"

    chunks = list(split_text(text))

    summarize_prompt = PromptTemplate(
        input_variables=["chunk"],
        template='"""{chunk}"""\n\nUsing the above text, please extract any key information and provide a detailed summary using bulletpoints. Do not include details about site navigation, menus, ads, etc. Only summarize the informative page content but do so as comprehensively as possible without leaving behind any important details.',
    )
    summarize_chain = get_chatgpt_chain(
        prompt=summarize_prompt, model_name="gpt-3.5-turbo", verbose=verbose
    )

    summaries = await asyncio.gather(
        *[summarize_chain.arun(chunk=chunk) for chunk in chunks]
    )
    combined_summary = "\n".join(summaries)

    if len(chunks) == 1 or not combine_summary:
        return combined_summary

    final_summary = await summarize_chain.arun(chunk=combined_summary)
    return final_summary


async def summarize_link(link: str) -> str:
    scraped_text = await scrape_text(link)
    summarized_text = await summarize_text(scraped_text)
    return f"Link Summary: {summarized_text}"
