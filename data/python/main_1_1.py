"""
TaskFlow main file
"""
from flask import Flask, request, jsonify
from taskflow_utils import create_task, get_task, update_task, delete_task
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///taskflow.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Models
class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(200))
    due_date = db.Column(db.String(20))
    priority = db.Column(db.String(10), default='Medium')

    def __repr__(self):
        return f"<Task {self.title}>"

# Routes
@app.route('/tasks', methods=['POST'])
def create_task_route():
    data = request.get_json()
    task = create_task(data)
    db.session.add(task)
    db.session.commit()
    return jsonify({'message': 'Task created successfully!', 'task': task.title}), 201

@app.route('/tasks/<int:task_id>', methods=['GET'])
def get_task_route(task_id):
    task = get_task(task_id)
    if task:
        return jsonify({'task': task.title, 'description': task.description, 'due_date': task.due_date, 'priority': task.priority}), 200
    return jsonify({'message': 'Task not found'}), 404

@app.route('/tasks/<int:task_id>', methods=['PUT'])
def update_task_route(task_id):
    data = request.get_json()
    task = update_task(task_id, data)
    db.session.commit()
    return jsonify({'message': 'Task updated successfully!', 'task': task.title}), 200

@app.route('/tasks/<int:task_id>', methods=['DELETE'])
def delete_task_route(task_id):
    task = delete_task(task_id)
    db.session.commit()
    return jsonify({'message': f'Task {task.title} deleted successfully'}), 200

if __name__ == '__main__':
    app.run(debug=True)
