import asyncio
import csv
import re

import aiohttp
from bs4 import BeautifulSoup
import requests

RENDER_API = "https://en.wiktionary.org/api/rest_v1/page/html"
WIKI_API = "https://en.wiktionary.org/w/api.php"
# API only allows sizes between 1-500
BATCH_SIZE = 500

# only allow modern Russian Cyrillic characters
# this also filters out whitespace to avoid compound words with adjectives that also decline
ALLOWED_TITLES = re.compile(
    r"^[АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя]+$",
)


async def get_declensions(session: aiohttp.ClientSession, title):
    async with session.get(f"{RENDER_API}/{title}") as response:
        tree = BeautifulSoup(await response.text())

        ru_section = tree.select_one("section:has(#Russian)")

        gen_sg = ru_section.select_one(".gen\|s-form-of")
        if gen_sg:
            gen_sg = gen_sg.text
        else:
            # no genitive means it's indeclinable
            return None

        gen_pl = ru_section.select_one(".gen\|p-form-of")
        if gen_pl:
            gen_pl = gen_pl.text
        else:
            # ditto
            return None

        return (gen_sg, gen_pl)


def get_nouns():
    session = requests.Session()
    params = {
        "action": "query",
        "cmtitle": "Category:Russian nouns",
        "cmlimit": BATCH_SIZE,
        "list": "categorymembers",
        "format": "json",
    }

    while True:
        response = session.get(WIKI_API, params=params)
        data = response.json()

        for member in data["query"]["categorymembers"]:
            word = member["title"]
            if ALLOWED_TITLES.fullmatch(word):
                yield word
            else:
                print(f"skipping complex word {word}")

        if continuation := data.get("continue"):
            params["cmcontinue"] = continuation["cmcontinue"]
        else:
            break


async def noun_data(session, noun):
    try:
        decl = await get_declensions(session, noun)
        if decl:
            print(f"parsed page for {noun}")
            return [noun, *decl]
        else:
            print(f"skipped indeclinable {noun}")
    except:
        return None


async def main():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for noun in get_nouns():
            tasks.append(noun_data(session, noun))
        result = await asyncio.gather(*tasks)

    with open("declensions.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(["title", "gen_sg", "gen_pl"])
        writer.writerows(
            row for row in result if row
        )  # filter out errors that became None


if __name__ == "__main__":
    asyncio.run(main())
