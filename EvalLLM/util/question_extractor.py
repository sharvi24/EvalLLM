import json
import re

from bs4 import BeautifulSoup
from urllib.request import urlopen


sample_url = "https://oll-resources.s3.us-east-2.amazonaws.com/oll3/store/titles/111/Plato_0131-01.html"
sample_start_of_text = "INTRODUCTION."
sample_patterns = [r"Jowett1892:\s*\d+", r"Edition: current; Page: \[\d+\]\s*"]


class QuestionExtractor:
    def __init__(
        self, url: str = sample_url, starting_point: str = sample_start_of_text
    ):
        self.url = url
        self.starting_point = starting_point

    def _clean_text(self, text, patterns: list = sample_patterns):
        for pattern in patterns:
            text = re.sub(pattern, "", text)
        return text

    def _extract_relevant_text(self):
        html = urlopen(self.url).read()
        soup = BeautifulSoup(html, features="html.parser")
        text = soup.get_text()
        split_text = text.split(self.starting_point)
        return split_text

    def read_text(self):
        stories = self._extract_relevant_text()
        stories = stories[1:]  # remove the first preface stuff
        for i, story in enumerate(stories):
            stories[i] = self._clean_text(text=story, patterns=sample_patterns)
        return stories

    def extract_context_question_answer(self, name, text):
        # limitation1: multi sentence answer
        # limitation2: What is that? said Critias.
        # maybe: With my consent? I said, or without my consent?

        # Extract all sentences where ? ends with \n
        pattern = r"[^.!?;]*\?\n"
        questions = re.findall(pattern, text)

        # Split the text based on the matches
        splits = re.split(pattern, text)

        # Remove empty strings from splits
        splits = [s.strip() for s in splits if s.strip()]

        # Make pairs of context, question, and answer
        pairs = []
        next_context = None
        for i in range(len(questions)):
            if not next_context:
                context = splits[i] if i < len(splits) else ""
            else:
                context = next_context
            question = questions[i]
            answer_and_next_context = splits[i + 1] if i + 1 < len(splits) else ""

            # Split the answer and next context based on the first "\n"
            answer, next_context = (
                answer_and_next_context.split("\n", 1)
                if "\n" in answer_and_next_context
                else (answer_and_next_context, "")
            )

            pairs.append({"context": context, "question": question, "answer": answer})

            # If there's remaining text after the first "\n," consider it as the next context
            if next_context:
                context = next_context.strip()

        file_name = f"prompts/vol1/{name}.json"

        # Write the JSON data to the file
        with open(file_name, "w") as json_file:
            json.dump(pairs, json_file, indent=4)

    def save_different_text(self):
        stories = self.read_text()
        for story in stories:
            story_name = story.split()[0]
            # file_path = f"prompts/vol1/{story_name}.txt"
            # with open(file_path, "w") as file:
                # file.write(story)
            self.extract_context_question_answer(name=story_name, text=story)
