import os
import pandas as pd
from openai import OpenAI

# Initialize the OpenAI client with your API key
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  # Ensure that OPENAI_API_KEY is set in your environment variables
)

# Load CSV and Excel to get categories
csv_path = 'Ophthal_FAQ2.csv'
excel_path = 'Ophthal_FAQ_Variations.xlsx'
csv_data = pd.read_csv(csv_path)
categories = pd.ExcelFile(excel_path).sheet_names

def categorize_question_answer(question, answer):
    # Format the message for the chat completion
    prompt = f"Question: {question}\nAnswer: {answer}\n\nWhich category does this question-answer pair belong to? {', '.join(categories)}"
    
    # Generate completion with the OpenAI client
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-3.5-turbo",  # Use an appropriate model that you have access to
    )
    
    # Extract the category from the response
    category = chat_completion.choices[0].message.content.strip()
    return category

# Process each row in the CSV file
results = []
for index, row in csv_data.iterrows():
    category = categorize_question_answer(row['questions'], row['answers'])
    results.append((row['questions'], row['answers'], category))

# Convert results to DataFrame and save to a new CSV file
labeled_data = pd.DataFrame(results, columns=['questions', 'answers', 'category'])
labeled_data.to_csv('categorized_qa_pairs.csv', index=False)

print("Categorization completed and saved to 'categorized_qa_pairs.csv'.")

