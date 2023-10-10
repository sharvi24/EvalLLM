import glob
import json
import math
import os


class PromptCreator:
    def __init__(self) -> None:
        pass

    def get_story_names(self, data_location: str = 'prompts/vol1'):
        story_files = glob.glob(f"{data_location}/*.json")
        story_names = [
            os.path.splitext(os.path.basename(sf))[0] for sf in story_files
        ]
        return story_names

    def get_stories_paths(self, data_location: str = 'prompts/vol1'):
        story_files = glob.glob(f"{data_location}/*.json")
        return story_files

    def create_prompts(self, data_location: str = 'prompts/vol1', max_seq_len=1024):
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
                prompt += f"Context:  {obj['context']}"
                f"Question: {obj['question']}"
                f"Answer: 1 Generate an Answer \n2 True or False \n"
                f"3 Given context is insufficient to answer this question."
                target = obj['answer']

                prompts.append(prompt)
                targets.append(target)
            prompts, targets = self.split_long_prompts(prompts, targets)
            all_prompts[story_name] = {"prompts": prompts, "targets": targets}
        return all_prompts
    
    def split_long_prompts(self, prompts, targets, max_seq_len=1024):
        split_prompts = []
        split_targets = []

        for prompt, target in zip(prompts, targets):
            if len(prompt) > max_seq_len:
                # Split the prompt
                num_parts = math.ceil(len(prompt) / max_seq_len)
                prompt_parts = [prompt[i:i+max_seq_len] for i in range(0, len(prompt), max_seq_len)]
                
                # Add the splits back with targets as None
                for part in prompt_parts:
                    split_prompts.append(part)
                    split_targets.append(None)  
            else:
                # Prompt is short enough, add original
                split_prompts.append(prompt)
                split_targets.append(target)

        return split_prompts, split_targets
