import argparse
import datetime
import json
import os
import torch

from EvalLLM import PromptCreator
from llama import Llama


def read_and_generate(args, prompt_creator, story_name: str = "Charmides.") -> str:
    # story_names = prompt_creator.get_story_names(data_location=args.prompts_dir)
    # print(f"story_names = {story_names}")

    all_prompts = prompt_creator.create_prompts(
        data_location=args.prompts_dir, max_seq_len=args.max_seq_len
    )
    # TODO: remove the below line, doing just for 1 experiment
    # story_names = ["Charmides."] Cratylus
    # ['Euthydemus.', 'Cratylus.', 'Lysis.', 'Symposium.', 'Laches.', 'Phaedrus.', 'Ion.', 'Charmides.', 'Protagoras.']
    story_names = [story_name]

    # create the results folder
    results_dir = os.path.join(
        "results/", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    )
    os.makedirs(results_dir)

    for story_name in story_names:
        print(f"Processing for story_name = {story_name}")
        prompts = all_prompts[story_name]["prompts"]
        targets = all_prompts[story_name]["targets"]
        generator = build_model(args)
        results = generate_with_llama(
            args=args, generator=generator, prompts=prompts, targets=targets
        )
        with open(f"{results_dir}/{story_name}.json", 'w') as f:
            json.dump(results, f)
        generator = None


def generate_with_llama(
    args,
    generator,
    prompts,
    targets,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_gen_len: int = 128,
):
    results = []
    for prompt, target in zip(prompts, targets):
        result = generator.text_completion(
            [prompt],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        results.append(
            {"prompt": prompt, "output": result[0]['generation'], "target": target}
        )
    return results


def build_model(args):
    torch.cuda.empty_cache()
    generator = Llama.build(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
    )
    return generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--prompts_dir',
        type=str,
        default='/scratch/sb7787/sharvi/EvalLLM/prompts/vol1',
    )
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--max_seq_len', type=int, default=4096)
    parser.add_argument('--max_gen_len', type=int, default=128)
    parser.add_argument('--max_batch_size', type=int, default=4)
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        default="/vast/work/public/ml-datasets/llama-2/llama-2-13b",
    )
    parser.add_argument(
        '--tokenizer_path',
        type=str,
        default="/vast/work/public/ml-datasets/llama-2/tokenizer.model",
    )
    parser.add_argument(
        '--story_name',
        type=str,
        default="Charmides.",
    )
    args = parser.parse_args()

    torch.cuda.empty_cache()

    prompt_creator = PromptCreator()
    read_and_generate(args, prompt_creator, args.story_name)


if __name__ == "__main__":
    main()


# from transformers import AutoTokenizer
# import transformers
# import torch

# model = "meta-llama/Llama-2-7b-chat-hf"
# # model = "/vast/work/public/ml-datasets/llama-2/llama-2-7b-chat"
# tokenizer = AutoTokenizer.from_pretrained(model)
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )

# sequences = pipeline(
#     'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
#     do_sample=True,
#     top_k=10,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
#     max_length=200,
# )
# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")
