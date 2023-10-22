import glob
import json
import math
import os
import re


class PromptCreator4GPT:
    def __init__(self) -> None:
        pass

    def get_story_names(self, data_location: str = 'dataset/vol1'):
        story_files = glob.glob(f"{data_location}/*.json")
        story_names = [
            os.path.splitext(os.path.basename(sf))[0] for sf in story_files
        ]
        return story_names

    def get_stories_paths(self, data_location: str = 'prompts/vol1'):
        story_files = glob.glob(f"{data_location}/*.json")
        return story_files

    def clean_text(self, text: str):
        # remove any words starting with \u
        regex = r"\\u\w+"
        text = re.sub(regex, "", text)
        text = text.replace("\n", "")
        return text

    def create_prompts_cot(
        self, data_location: str = 'dataset/vol1', max_seq_len=100000
    ):
        story_files = self.get_stories_paths(data_location=data_location)
        all_prompts = {}
        for story_file in story_files:
            story_name, _ = os.path.splitext(os.path.basename(story_file))
            all_prompts[story_name] = {}
            prompts, targets = [], []
            with open(story_file) as f:
                data = json.load(f)

            for obj in data:
                prompt, target = "", ""
                context = obj["context"]
                context = self.clean_text(context)

                question = obj["question"]
                question = self.clean_text(question)

                if len(context) > max_seq_len:
                    # Dividing the context to make it smaller, and make targets None
                    context = (
                        "I will give you a long passage in parts, answer the question at end. Passage: "
                        + context
                    )
                    split_prompts, split_targets = self.split_long_prompts(
                        context, max_seq_len
                    )
                    prompts.extend(split_prompts)
                    targets.extend(split_targets)
                else:
                    prompt += f"I will give you a passage and a question, you are an expert in this, please provide a precise answer. Passage: {context} "

                prompt += f" Question: {question}"
                prompt += " Do you think the given passage is sufficient to answer this question, yes or no?"
                prompt += " If yes, first give step-by-step reasoning about how to answer the question. Then output the answer."
                target = obj['answer']
                prompts.append(prompt)
                targets.append(target)

            all_prompts[story_name] = {"prompts": prompts, "targets": targets}
        return all_prompts

    def split_long_prompts(self, context, max_seq_len=1024):
        split_prompts = []
        split_targets = []

        # Split the context
        num_parts = math.ceil(len(context) / max_seq_len)
        prompt_parts = [
            context[i : i + max_seq_len] for i in range(0, len(context), max_seq_len)
        ]

        # Add the splits back with targets as None
        for part in prompt_parts:
            split_prompts.append(part)
            split_targets.append(None)

        return split_prompts, split_targets
