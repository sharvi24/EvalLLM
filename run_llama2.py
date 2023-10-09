import argparse
import os
import torch

from EvalLLM import PromptCreator
from llama.llama import Llama


def read_and_generate(args, prompt_creator) -> str:
    story_names = prompt_creator.get_story_names(data_location=args.prompts_dir)
    all_prompts = prompt_creator.create_prompts(data_location=args.prompts_dir)
    for story_name in story_names:
        prompts = all_prompts[story_name]["prompts"]
        targets = all_prompts[story_name]["targets"]
        generator = build_model(args)
        generate_with_llama(generator=generator, prompts=prompts, targets=targets)

        # prompts = []
        # full_path = os.path.join(directory_path, csv_file)
        # with open(full_path, mode='r') as file:
        #     csv_reader = csv.reader(file)
        #     for row in csv_reader:
        #         output = f"Text: {row[1]}, Question: {row[2]}\n"
        #         prompts.append(output)
        # print(len(prompts))
        # final_path = f"/scratch/sb7787/duygu/llama-results/{csv_file[:-4]}"


def generate_with_llama(
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
        results.append({"output": result[0]['generation'], "target": target})
        results.append("==================================")
        break

    print(results)
    # import numpy as np

    # np.savetxt(f"{csv_file_name}-results.csv", results, delimiter=",", fmt='%s')


def build_model(args):
    generator = Llama.build(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
    )
    return generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts_dir', type=str, default='prompts/vol1')
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--max_seq_len', type=int, default=4096)
    parser.add_argument('--max_gen_len', type=int, default=128)
    parser.add_argument('--max_batch_size', type=int, default=4)
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        default="/vast/work/public/ml-datasets/llama-2/llama-2-7b",
    )
    parser.add_argument(
        '--tokenizer_path',
        type=str,
        default="/vast/work/public/ml-datasets/llama-2/tokenizer.model",
    )
    args = parser.parse_args()

    torch.cuda.empty_cache()

    prompt_creator = PromptCreator()
    read_and_generate(args, prompt_creator)


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
