import re


def clean_text(text):
    text = text.replace("“", "'")
    text = text.replace("”", "'")
    text = text.replace("‘", "'")
    text = text.replace("’", "'")
    text = re.sub("■\s?", "", text)
    text = re.sub("①\s?", "", text)
    text = re.sub("▶\s?", "", text)
    text = re.sub("▽\s?", "", text)
    text = re.sub("○\s?", "", text)
    text = re.sub("◇\s?", "", text)
    text = re.sub("―\s?", "", text)
    text = re.sub("◆\s?", "", text)
    text = re.sub("▼\s?", "", text)
    text = re.sub("△\s?", "", text)


    return text