from flask import Flask, render_template, request
import os

app = Flask(__name__)

# Study material structure
TOPICS = {
    'statistics': {
        'name': 'Statistics',
        'icon': '📊',
        'color': '#667eea',
        'subtopics': [
            {'id': 'descriptive-stats', 'title': 'Descriptive Statistics',
             'description': 'Mean, median, mode, variance'},
            {'id': 'probability', 'title': 'Probability Theory',
             'description': 'Basics of probability and distributions'},
            {'id': 'hypothesis-testing', 'title': 'Hypothesis Testing', 'description': 't-test, chi-square, ANOVA'},
            {'id': 'regression-analysis', 'title': 'Regression Analysis',
             'description': 'Linear and logistic regression'},
            {'id': 'distributions', 'title': 'Probability Distributions', 'description': 'Normal, binomial, poisson'},
        ]
    },
    'machine-learning': {
        'name': 'Machine Learning',
        'icon': '🤖',
        'color': '#f093fb',
        'subtopics': [
            {'id': 'supervised-learning', 'title': 'Supervised Learning',
             'description': 'Classification and regression'},
            {'id': 'unsupervised-learning', 'title': 'Unsupervised Learning',
             'description': 'Clustering and dimensionality reduction'},
            {'id': 'model-evaluation', 'title': 'Model Evaluation',
             'description': 'Metrics, cross-validation, confusion matrix'},
            {'id': 'feature-engineering', 'title': 'Feature Engineering',
             'description': 'Selection, extraction, transformation'},
            {'id': 'ensemble-methods', 'title': 'Ensemble Methods', 'description': 'Random Forest, XGBoost, Stacking'},
        ]
    },
    'deep-learning': {
        'name': 'Deep Learning',
        'icon': '🧠',
        'color': '#4facfe',
        'subtopics': [
            {'id': 'neural-networks', 'title': 'Neural Networks Basics',
             'description': 'Perceptrons, activation functions'},
            {'id': 'cnn', 'title': 'Convolutional Neural Networks',
             'description': 'Image processing and computer vision'},
            {'id': 'rnn-lstm', 'title': 'RNN & LSTM', 'description': 'Sequential data and time series'},
            {'id': 'optimization', 'title': 'Optimization Techniques',
             'description': 'SGD, Adam, learning rate scheduling'},
            {'id': 'reduce-lr-plateau', 'title': 'ReduceLROnPlateau',
             'description': 'Understanding plateaus and learning rate reduction'},
            {'id': 'transfer-learning', 'title': 'Transfer Learning',
             'description': 'Pre-trained models and fine-tuning'},
        ]
    },
    'dsa': {
        'name': 'Data Structures & Algorithms',
        'icon': '🔢',
        'color': '#43e97b',
        'subtopics': [
            {'id': 'arrays-strings', 'title': 'Arrays & Strings', 'description': 'Basic data structures'},
            {'id': 'linked-lists', 'title': 'Linked Lists', 'description': 'Single, double, circular'},
            {'id': 'stacks-queues', 'title': 'Stacks & Queues', 'description': 'LIFO and FIFO structures'},
            {'id': 'trees', 'title': 'Trees', 'description': 'Binary trees, BST, AVL'},
            {'id': 'graphs', 'title': 'Graphs', 'description': 'BFS, DFS, shortest path'},
            {'id': 'sorting', 'title': 'Sorting Algorithms', 'description': 'Quick, merge, heap sort'},
            {'id': 'dynamic-programming', 'title': 'Dynamic Programming', 'description': 'Memoization and tabulation'},
        ]
    },
    'python': {
        'name': 'Python Programming',
        'icon': '🐍',
        'color': '#fa709a',
        'subtopics': [
            {'id': 'python-basics', 'title': 'Python Basics', 'description': 'Variables, data types, control flow'},
            {'id': 'oop', 'title': 'Object-Oriented Programming', 'description': 'Classes, inheritance, polymorphism'},
            {'id': 'numpy-pandas', 'title': 'NumPy & Pandas', 'description': 'Data manipulation libraries'},
            {'id': 'decorators', 'title': 'Decorators & Generators', 'description': 'Advanced Python concepts'},
            {'id': 'async', 'title': 'Async Programming', 'description': 'asyncio and concurrent programming'},
        ]
    },
    'sql': {
        'name': 'SQL & Databases',
        'icon': '🗄️',
        'color': '#feca57',
        'subtopics': [
            {'id': 'sql-basics', 'title': 'SQL Basics', 'description': 'SELECT, INSERT, UPDATE, DELETE'},
            {'id': 'joins', 'title': 'SQL Joins', 'description': 'INNER, LEFT, RIGHT, FULL joins'},
            {'id': 'aggregations', 'title': 'Aggregations & Groups',
             'description': 'GROUP BY, HAVING, aggregate functions'},
            {'id': 'indexes', 'title': 'Indexing & Performance', 'description': 'Query optimization'},
            {'id': 'transactions', 'title': 'Transactions & ACID', 'description': 'Database consistency'},
        ]
    }
}


@app.route('/')
def index():
    return render_template("index.html",topics=TOPICS)


@app.route('/topic/<topic_id>')
def topic_detail(topic_id):
    if topic_id in TOPICS:
        return render_template('topic.html', topic_id=topic_id, topic=TOPICS[topic_id])
    return "Topic not found", 404


@app.route('/content/<topic_id>/<subtopic_id>')
def content_detail(topic_id, subtopic_id):
    if topic_id in TOPICS:
        topic = TOPICS[topic_id]
        subtopic = next((s for s in topic['subtopics'] if s['id'] == subtopic_id), None)
        if subtopic:
            # Check if this is the reduce-lr-plateau topic
            if subtopic_id == 'reduce-lr-plateau':
                return render_template('reduce_lr_plateau.html', topic=topic, subtopic=subtopic, topic_id=topic_id)
            return render_template('content.html', topic=topic, subtopic=subtopic, topic_id=topic_id)
    return "Content not found", 404


@app.route('/search')
def search():
    query = request.args.get('q', '').lower()
    results = []

    for topic_id, topic in TOPICS.items():
        for subtopic in topic['subtopics']:
            if query in subtopic['title'].lower() or query in subtopic['description'].lower():
                results.append({
                    'topic_name': topic['name'],
                    'topic_id': topic_id,
                    'subtopic': subtopic
                })

    return render_template('search.html', query=query, results=results)


if __name__ == '__main__':
    app.run()

