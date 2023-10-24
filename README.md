# EvalLLM


## Prompts

### Zero-shot prompting on reasoning benchmark

```
I will give you a passage and a question, you are an expert in this, please provide a precise answer. Passage: <passage>  Question: <question> Do you think the given passage is sufficient to answer this question, yes or no? If yes, first give step-by-step reasoning about how to answer the question. Then output the answer."
```

### Evaluation prompts
Self-ask GPT3.5
```
Given is target answer and response (reasoning + answer in last para). You have to compare the response with the target and check if the response matches with the actual answer (target). Provide your response as json with 2 fields "reasons" and "match-percent".  Note: if the response is that the passage is insufficient, then "match-percent" should be negative.

"target": <target>
"response": <response>
```