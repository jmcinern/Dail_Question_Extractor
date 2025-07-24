from tqdm import tqdm
import fasttext
import os

# Path to your Dáil text file
dail_file_path = "C:/Users/josep/VS-code-projects/CPT-Dáil/data/dail.txt"

with open(dail_file_path, "r", encoding="utf-8") as f:
    dail_text = f.read()

def get_utterences(text):
    utterences = text.split("<|endoftext|>")
    return [utterence.strip() for utterence in utterences if utterence.strip()]

sentences = get_utterences(dail_text)

# Get questions, utterances marked with a question mark
questions = [sentence for sentence in sentences if sentence and sentence[-1] == "?"]

print("Proportion of questions in the text:", len(questions) / len(sentences))
print(len(questions), "questions found in the text.")

# Load fastText language identification model
FASTTEXT_MODEL_PATH = "lid.176.bin"
if not os.path.exists(FASTTEXT_MODEL_PATH):
    import urllib.request
    print("Downloading fastText language ID model...")
    url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    urllib.request.urlretrieve(url, FASTTEXT_MODEL_PATH)

ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)

irish_questions = []
english_questions = []

def detect_language(text):
    prediction = ft_model.predict(text.replace('\n', ' '), k=1)
    lang = prediction[0][0].replace("__label__", "")
    return lang

for question in tqdm(questions, desc="Detecting language"):
    lang = detect_language(question)
    if lang == "ga":
        irish_questions.append(question)
    elif lang == "en":
        english_questions.append(question)
    # else: ignore other languages

print(f"Found {len(irish_questions)} Irish questions and {len(english_questions)} English questions.")

# Save question lists to files, separating with new line
with open("irish_questions.txt", "w", encoding="utf-8") as f:
    for question in irish_questions:
        f.write(question + "\n")

with open("english_questions.txt", "w", encoding="utf-8") as f:
    for question in english_questions:
        f.write(question + "\n")