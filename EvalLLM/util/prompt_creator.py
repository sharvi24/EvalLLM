import os
import json
import glob


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

    def create_prompts(self, data_location: str = 'prompts/vol1'):
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
                prompt += f"context:  {obj['context']}"
                f"question: {obj['question']}"
                f"answer: 1 Generate an Answer \n2 True or False \n"
                f"3 Given context is insufficient to answer this question."
                target = obj['answer']

                prompts.append(prompt)
                targets.append(target)
            all_prompts[story_name] = {"prompts": prompts, "targets": targets}
        return all_prompts
