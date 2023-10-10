import json
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import wordnet
from fuzzywuzzy import fuzz
nltk.download('wordnet')
with open('results/Charmides..json') as f:
  results = json.load(f)

type1_count = 0
type2_count = 0
type3_count = 0

for res in results:
  if res['target'] is None:
    continue
  
  if fuzz.partial_ratio(res['output'], res['target']) > 80:
    type3_count += 1
    
  else:
    target_synsets = wordnet.synsets(res['target'])

    for syn in target_synsets:
      for lemma in syn.lemmas():
        if lemma.name() == res['output']:
          type2_count += 1
          break

    if type2_count == 0:  
      if fuzz.partial_ratio(res['output'], res['target']) > 60:
        type1_count += 1

# Visualizations  
plt.bar(['Type 1', 'Type 2', 'Type 3'], [type1_count, type2_count, type3_count])
plt.title('Output Type Counts')
plt.ylabel('Count')
plt.savefig('output_types.png')

total = type1_count + type2_count + type3_count
accuracy = type1_count / total
print(f'Accuracy: {accuracy*100:.2f}%')