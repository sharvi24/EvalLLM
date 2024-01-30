# EvalLLM


## High Level design
![alt text](scripts/figures/EvalLLM_high_level_design.png?raw=true)

## Prompts

### Zero-shot prompting on reasoning benchmark

```
I will give you a passage and a question, you are an expert in this, please provide a precise answer. 
Passage: <passage>  
Question: <question> 
Do you think the given passage is sufficient to answer this question, yes or no? 
If yes, first give step-by-step reasoning about how to answer the question. Then output the answer."
```

### Generate FOL
```
"""

prover9 from nltk requires the first order logic in a different way than normally written FOL, refer the below code to understand.

>>> from nltk.inference import *
>>> from nltk.sem import LogicParser, ApplicationExpression
>>> lp = LogicParser()
>>> bicond = lp.parse('(exists x.(man(x) and walks(x)) <-> exists x.(walks(x) and man(x)))')
>>> get_prover(bicond, prover_name='tableau').prove()
True


Generate first order logic for the given statement in the same way so that it can be directly parsed by nltk. 
I don't need the code, I just need the First order logic statement.

Statement: <statement>

FOL:

"""
```

### Evaluation prompts
Self-ask GPT3.5
```
Given is target answer and response (reasoning + answer in last para). 
You have to compare the response with the target and check if the response matches with the actual answer (target). 
Provide your response as json with 2 fields "reasons" and "match-percent".  
Note: if the response is that the passage is insufficient, then "match-percent" should be negative.

"target": <target>
"response": <response>
```