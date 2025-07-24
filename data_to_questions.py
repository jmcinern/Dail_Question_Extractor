from tqdm import tqdm
import os
import json
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_anthropic import ChatAnthropic
# read file
# tkn by sentences
# identify questions

dail_file_path = "/users/40460549/sharedscratch/questions_dail_ext/Dail_Question_Extractor/irish_non_questions.txt"

with open(dail_file_path, "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]


def get_anthropic_api_key():
    """Get Anthropic API key from environment or secrets file"""
    # Try environment variable first
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        return api_key

    # Try secrets file as fallback
    try:
        with open("secrets.json", "r", encoding="utf-8") as f:
            secrets = json.load(f)[0]
            return secrets["anthropic"]
    except FileNotFoundError:
        raise FileNotFoundError(
            "No Anthropic API key found" 
        )

CLAUDE_MODEL = "claude-3-5-haiku-20241022"  # Updated to the correct model

def create_claude_instance():
    """Create a new Claude instance"""
    return ChatAnthropic(
        model=CLAUDE_MODEL,
        temperature=0.9,
        api_key=get_anthropic_api_key(),
    )

# Load examples from JSON
with open("./examples.json", "r", encoding="utf-8") as f:
    examples = json.load(f)

# Create example prompt template
example_prompt = ChatPromptTemplate.from_messages(
    [("human", "statement: {statement}"), ("assistant", "{question}")],
)

# Create few-shot prompt with examples
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# Load system message
with open("./system.txt", "r", encoding="utf-8") as f:
    system_message = f.read()

# Create the full prompt template
final_prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    few_shot_prompt,
    ("human", "statement: {statement}")
])


claude = create_claude_instance()
# Generate responses
Q_A = []  # Fixed: removed empty tuple
for A in tqdm(not_questions[:5], desc="Generating responses"):
    try:
        formatted_prompt = final_prompt.format_messages(statement=A)
        response = claude.invoke(formatted_prompt)
        Q = response.content
        Q_A.append({"statement": A, "question": Q})  
    except Exception as e:
        print(f"Error processing statement '{A[:50]}...': {e}")
        continue

# Save to file
output_path = "C:/Users/josep/VS-code-projects/DPO_Synth/outputs/questinos_synth.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create directory if it doesn't exist

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(Q_A, f, ensure_ascii=False, indent=4)

print(f"Successfully generated {len(Q_A)} question-answer pairs")
print(f"Output saved to: {output_path}")


