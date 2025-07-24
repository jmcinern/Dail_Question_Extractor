from tqdm import tqdm
import pandas as pd
import re
# read file
# tkn by sentences
# identify questions

dail_file_path = "C:/Users/josep/VS-code-projects/CPT-Dáil/data/dail.txt"

with open(dail_file_path, "r", encoding="utf-8") as f:
    dail_text = f.read()


def get_utterences(text):
    utterences = text.split("<|endoftext|>")
    return [utterence.strip() for utterence in utterences if utterence.strip()]

sentences = get_utterences(dail_text)

# get questions, utternces marked with a question mark
questions = []
for sentence in sentences:
    if sentence[-1] == "?":
        questions.append(sentence)

print("Proportion of questions in the text:", len(questions) / len(sentences))
print(len(questions), "questions found in the text.")
# Identfy whether in Irish or English
irish_questions = []
english_questions = []

words_by_freq_tsv_path = "C:/Users/josep/Downloads/cng-lem.tsv/cng-lem.tsv" 
words_freq_df = pd.read_csv(words_by_freq_tsv_path, sep="\t", header=None)
top_200_words_ga = words_freq_df.iloc[:200, 0].tolist()

# print(top_200_words_ga), remove: ['an', 'a', 'is', 'as', 'do', 'go', 'sin', 'am', 'eh']

# remove overlapping en/ga words
top_200_words_ga = [word for word in top_200_words_ga if word not in ["an", "a", "is", "as", "do", "go", "sin", "am", "eh"]]


def is_Irish(text, top_200_word_gas):
    for kw_ga in top_200_words_ga:
        # Match whole word only
        if re.search(r'\b' + re.escape(kw_ga) + r'\b', text) and "the " not in text:
            return True
    return False

for question in tqdm(questions):
    if is_Irish(question, top_200_words_ga):
        irish_questions.append(question)
    else:
        english_questions.append(question)

print(f"Found {len(irish_questions)} Irish questions and {len(english_questions)} English questions.")

# save question lists to files, separating with new line
with open("irish_questions.txt", "w", encoding="utf-8") as f:
    for question in irish_questions:
        f.write(question + "\n")

with open("english_questions.txt", "w", encoding="utf-8") as f:
    for question in english_questions:
        f.write(question + "\n")




