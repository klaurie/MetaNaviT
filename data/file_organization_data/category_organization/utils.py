"""
TaskFlow utils file
"""
from models import Task
from datetime import datetime

# Function to create a task
def create_task(data):
    title = data.get('title')
    description = data.get('description', '')
    due_date = data.get('due_date', None)
    priority = data.get('priority', 'Medium')

    # Validate data
    if not title:
        raise ValueError("Title is required")
    
    task = Task(title=title, description=description, due_date=due_date, priority=priority)
    return task

# Function to retrieve a task
def get_task(task_id):
    task = Task.query.get(task_id)
    return task

# Function to update a task
def update_task(task_id, data):
    task = Task.query.get(task_id)
    if not task:
        raise ValueError("Task not found")
    
    task.title = data.get('title', task.title)
    task.description = data.get('description', task.description)
    task.due_date = data.get('due_date', task.due_date)
    task.priority = data.get('priority', task.priority)
    
    return task

# Function to delete a task
def delete_task(task_id):
    task = Task.query.get(task_id)
    if not task:
        raise ValueError("Task not found")
    
    Task.query.filter_by(id=task_id).delete()
    return task
