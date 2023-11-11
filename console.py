import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
import datetime

# Sample tasks
tasks = [
    {"id": 1, "task": "Finish report on AI", "due_date": "2023-12-01"},
    {"id": 2, "task": "Buy groceries", "due_date": "2023-11-15"},
    # Add more tasks as needed
]

# Function to tokenize and vectorize text
def tokenize_and_vectorize(text):
    tokens = word_tokenize(text.lower())
    return ' '.join(tokens)

# Function to get similarity score between two texts
def get_similarity_score(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(vectors)
    return similarity[0][1]

# Function to add a task to the to-do list
def add_task(task_text, due_date):
    task_id = len(tasks) + 1
    tasks.append({"id": task_id, "task": task_text, "due_date": due_date})
    return f"Task added: {task_text}"

# Function to recommend tasks based on user input
def recommend_tasks(user_input):
    user_input = tokenize_and_vectorize(user_input)

    # Calculate similarity scores between user input and existing tasks
    scores = [(task["task"], get_similarity_score(user_input, tokenize_and_vectorize(task["task"])))
              for task in tasks]

    # Sort tasks by similarity score in descending order
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Filter out tasks with low similarity scores
    recommended_tasks = [task[0] for task in scores if task[1] > 0.5]

    return recommended_tasks

# Function to mark a task as completed
def complete_task(task_id):
    for task in tasks:
        if task["id"] == task_id:
            tasks.remove(task)
            return f"Task completed: {task['task']}"

# Function to display the to-do list
def display_todo_list():
    print("To-Do List:")
    for task in tasks:
        print(f"{task['id']}. {task['task']} (Due: {task['due_date']})")

# Sample user interactions
print(add_task("Read a book", "2023-11-20"))
print(add_task("Work on coding project", "2023-11-25"))
print(add_task("Call mom", "2023-11-13"))

user_input = "Finish report on artificial intelligence"
recommended_tasks = recommend_tasks(user_input)
print(f"Recommended tasks: {recommended_tasks}")

display_todo_list()

task_to_complete = 2  # Replace with the actual task ID to mark as completed
print(complete_task(task_to_complete))

display_todo_list()

