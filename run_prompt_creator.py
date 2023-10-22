import json
from EvalLLM import PromptCreator4GPT

DATA_DIR = "dataset/vol1"
PROMPTS_DIR = "dataset/prompts/gpt-cot"


def main():
    pc = PromptCreator4GPT()
    all_prompts = pc.create_prompts_cot(data_location=DATA_DIR)
    for story_name, d in all_prompts.items():
        story_name = story_name[:-1]  # remove the dot
        res = []
        for prompt, target in zip(d["prompts"], d["targets"]):
            res.append({"prompt": prompt, "target": target, "response": ""})
        with open(f"{PROMPTS_DIR}/{story_name}.json", 'w') as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()
