import json
import matplotlib.pyplot as plt
import os
from evaluate import load
from rouge_score import rouge_scorer
from statistics import mean

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
bertscore = load("bertscore")
path = "results/2023-10-14"
story_files = os.listdir(path)

for k, story_file in enumerate(story_files):
    print(f"Evaluating for story_file = {story_file}")
    prompts = []
    model_outputs = []
    model_str_outputs = []
    ground_truth = []
    ground_str_truth = []
    scores = []
    rouge_scores = []

    with open('results/2023-10-14/' + story_file) as f:
        results = json.load(f)

    for res in results:
        if res['target'] is None:
            continue
        prompts.append([res['prompt']])
        model_outputs.append([res['output']])
        ground_truth.append([res['target']])
        model_str_outputs.append(res['output'])
        ground_str_truth.append(res['target'])

    for i, model_output in enumerate(model_outputs):
        score = bertscore.compute(
            predictions=model_output, references=ground_truth[i], lang="en"
        )
        scores.append(score)

        rouge_score = scorer.score(model_str_outputs[i], ground_str_truth[i])
        rouge_scores.append(rouge_score)

    avg_f1_score = mean([s['f1'][0] for s in scores])
    avg_precision_score = mean([s['precision'][0] for s in scores])
    avg_recall_score = mean([s['recall'][0] for s in scores])
    avg_rouge1_f1_score = mean([p['rouge1'][2] for p in rouge_scores])
    avg_rouge1_precision_score = mean([p['rouge1'][0] for p in rouge_scores])
    avg_rouge1_recall_score = mean([p['rouge1'][1] for p in rouge_scores])

    avg_rougeL_f1_score = mean([t['rougeL'][2] for t in rouge_scores])
    avg_rougeL_precision_score = mean([t['rougeL'][0] for t in rouge_scores])
    avg_rougeL_recall_score = mean([t['rougeL'][1] for t in rouge_scores])

    # Visualizations
    plt.bar(
        ['F1_Score', 'Precision', 'Recall'],
        [avg_f1_score, avg_precision_score, avg_recall_score],
    )
    plt.title(f"Average BERT Scores for {story_file}")
    plt.savefig(f'EvaluationResults/BERTscores-{story_file}.png')
    plt.clf()

    # Visualizations
    plt.bar(
        ['F1_Score', 'Precision', 'Recall'],
        [avg_rouge1_f1_score, avg_rouge1_precision_score, avg_rouge1_recall_score],
    )
    plt.title(f"Average ROUGE1 Scores for {story_file}")
    plt.savefig(f"EvaluationResults/ROUGEscores - {story_file}.png")
    plt.clf()

    # Visualizations
    plt.bar(
        ['F1_Score', 'Precision', 'Recall'],
        [avg_rougeL_f1_score, avg_rougeL_precision_score, avg_rougeL_recall_score],
    )
    plt.title(f"Average ROUGEL Scores for {story_file}")
    plt.savefig(f'EvaluationResults/ROUGELscores - {story_file}.png')
