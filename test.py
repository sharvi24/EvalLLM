# from urllib.request import urlopen
# from bs4 import BeautifulSoup
# import re
# sample_url = "https://oll-resources.s3.us-east-2.amazonaws.com/oll3/store/titles/111/Plato_0131-01.html"
# start_of_text = "INTRODUCTION"

# html = urlopen(sample_url).read()
# soup = BeautifulSoup(html, features="html.parser")

# text = soup.get_text()
# split_text = text.split('INTRODUCTION.')
# stories = split_text[1:]  # remove the first preface stuff
# for story in stories:
#     story_name = story.split()[0]
#     story_content = story

#     # Specify the file path
#     file_path = f"prompts/vol1/{story_name}.txt"

#     # Open the file in write mode
#     with open(file_path, "w") as file:
#         # Write the content to the file
#         file.write(story_content)

# # print(split_text[1].split()[0])
# # print(split_text[1].split()[1:])
# # relevant_text = split_text[0]
# # print(relevant_text)
import re
import json


def extract_context_question_answer(text):
    # Extract all sentences where ? ends with \n
    pattern = r'[^.!?;]*\?\n'
    questions = re.findall(pattern, text)

    # Split the text based on the matches
    splits = re.split(pattern, text)

    # Remove empty strings from splits
    splits = [s.strip() for s in splits if s.strip()]

    # Make pairs of context, question, and answer
    pairs = []
    next_context = None
    for i in range(len(questions)):
        if not next_context:
            context = splits[i] if i < len(splits) else ""
        else:
            context = next_context
        question = questions[i]
        answer_and_next_context = splits[i + 1] if i + 1 < len(splits) else ""

        # Split the answer and next context based on the first "\n"
        answer, next_context = answer_and_next_context.split(
            '\n', 1) if '\n' in answer_and_next_context else (answer_and_next_context, "")

        pairs.append({
            "context": context,
            "question": question,
            "answer": answer
        })

        # If there's remaining text after the first "\n," consider it as the next context
        if next_context:
            context = next_context.strip()

    file_name = "prompts/vol1/output.json"

    # Write the JSON data to the file
    with open(file_name, 'w') as json_file:
        json.dump(pairs, json_file)
   # json_string = json.dumps(pairs)

    # Print the pairs
    for context, question, answer in pairs:
        print("Context:", context)
        print("Question:", question)
        print("Answer:", answer)
        print()


# Example usage:
text = """
This is the context sentence. questions within context? Is this the first question?
Yes, this is the answer to the first question.
Now, another context. Is this the second question?
No, this is the answer to the second question.
Now, another context. Is this the third question?
No, this is the answer to the third question.
"""

extract_context_question_answer(text)
