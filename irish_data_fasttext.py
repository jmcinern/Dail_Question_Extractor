from tqdm import tqdm
import fasttext
import os

# Path to your Dáil text file
dail_file_path = "/users/40460549/sharedscratch/cpt-dail/data/dail.txt"

with open(dail_file_path, "r", encoding="utf-8") as f:
    dail_text = f.read()

def get_utterences(text):
    utterences = text.split("<|endoftext|>")
    return [utterence.strip() for utterence in utterences if utterence.strip()]

sentences = get_utterences(dail_text)

# Get non-questions (utterances NOT ending with a question mark)
non_questions = [sentence for sentence in sentences if sentence and sentence[-1] != "?"]

print("Proportion of non-questions in the text:", len(non_questions) / len(sentences))
print(len(non_questions), "non-questions found in the text.")

# Load fastText language identification model
FASTTEXT_MODEL_PATH = "lid.176.bin"
if not os.path.exists(FASTTEXT_MODEL_PATH):
    import urllib.request
    print("Downloading fastText language ID model...")
    url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    urllib.request.urlretrieve(url, FASTTEXT_MODEL_PATH)

ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)

irish_non_questions = []
english_non_questions = []

def detect_language(text):
    prediction = ft_model.predict(text.replace('\n', ' '), k=1)
    lang = prediction[0][0].replace("__label__", "")
    return lang

for sentence in tqdm(non_questions, desc="Detecting language"):
    lang = detect_language(sentence)
    if lang == "ga":
        irish_non_questions.append(sentence)
    elif lang == "en":
        english_non_questions.append(sentence)
    # else: ignore other languages

print(f"Found {len(irish_non_questions)} Irish non-questions and {len(english_non_questions)} English non-questions.")

# Save non-question lists to files, separating with new line
with open("irish_non_questions.txt", "w", encoding="utf-8") as f:
    for sentence in irish_non_questions:
        f.write(sentence + "\n")

with open("english_non_questions.txt", "w", encoding="utf-8") as f:
    for sentence in english_non_questions:
        f.write(sentence + "\n")