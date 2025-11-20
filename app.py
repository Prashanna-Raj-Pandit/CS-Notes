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
            {'id': 'bias-variance', 'title': 'Bias Variance Tradeoff',
             'description': 'Understanding of Under fitting (Bias) and Overfitting(Variance)'},
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
             'description': 'Perceptrons, activation functions, backpropagation'},
            {'id': 'optimizers', 'title': 'Optimizers',
             'description': 'Different types of optimizers'},
            {'id': 'cnn', 'title': 'Convolutional Neural Networks',
             'description': 'Image processing and computer vision'},
            {'id': 'rnn-lstm', 'title': 'RNN & LSTM', 'description': 'Sequential data and time series'},
            {'id': 'optimization', 'title': 'Optimization Techniques',
             'description': 'SGD, Adam, learning rate scheduling'},
            {'id': 'reduce-lr-plateau', 'title': 'ReduceLROnPlateau',
             'description': 'Understanding plateaus and learning rate reduction'},
            {'id': 'reproducibility', 'title': 'Reproducibility',
             'description': 'From randomness to determinism in TensorFlow'},
            {'id': 'transfer-learning', 'title': 'Transfer Learning',
             'description': 'Pre-trained models and fine-tuning'},
        ]
    },
    'dpp': {
        'name': 'Data Preprocessing',
        'icon': '🔢',
        'color': '#43e97b',
        'subtopics': [
            {'id': 'interpolation', 'title': 'Interpolation and Imputation',
             'description': 'Understanding of filling missing values.'},
            {'id': 'savgol', 'title': 'Savitzky-Golay Filter',
             'description': 'Understanding of Savitzky-Golay Filter, Polyorder, Window.'}
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
    'ts': {
        'name': 'Time Series',
        'icon': '📈',
        'color': '#feca57',
        'subtopics': [
            {'id': 'seg-emb', 'title': 'Time series segmentation and Embeddings', 'description':
                'Segmentation -> Normalization-> Embedding ->classify'},
            {'id': 'tsmote', 'title': 'T-SMOTE', 'description':
                'Understanding of T-SMOTE and SMOTE vs T-SMOTE'},

            {'id': 'ts-modeling', 'title': 'Time series Modeling', 'description':
                'visualization, 1D Conv, Local Pattern Detection,Dimensionality Reduction'},
            {'id': 'hybrid_model', 'title': 'Hybrid ASD Classification Pipeline',
             'description': 'Understanding of Hybrid Multi-View ASD Classification Pipeline'},

        ]
    },
    'dsa': {
        'name': 'Data Structures',
        'icon': '⌗',
        'color': '#43e97b',
        'subtopics': [
            {'id': 'linked-lists', 'title': 'Linked Lists', 'description': 'Single, double, circular'},
            {'id': 'stacks-queues', 'title': 'Stacks & Queues', 'description': 'LIFO and FIFO structures'},
            {'id': 'trees', 'title': 'Trees', 'description': 'Binary trees, BST, AVL'},
            {'id': 'graphs', 'title': 'Graphs', 'description': 'BFS, DFS, shortest path'},
            {'id': 'sorting', 'title': 'Sorting Algorithms', 'description': 'Quick, merge, heap sort'},
            {'id': 'dynamic-programming', 'title': 'Dynamic Programming', 'description': 'Memoization and tabulation'},
        ]
    },
}


@app.route('/')
def index():
    return render_template("index.html", topics=TOPICS)


@app.route('/topic/<topic_id>')
def topic_detail(topic_id):
    if topic_id in TOPICS:
        return render_template('topic.html', topic_id=topic_id, topic=TOPICS[topic_id])
    return "Topic not found", 404


@app.route('/content/<topic_id>/<subtopic_id>')
def content_detail(topic_id, subtopic_id):
    if topic_id in TOPICS:
        topic = TOPICS[topic_id]
        # print(topic)
        subtopic = next((s for s in topic['subtopics'] if s['id'] == subtopic_id), None)
        print(subtopic)
        if subtopic:
            # Check if this is the reduce-lr-plateau topic
            if subtopic_id == 'reduce-lr-plateau':
                return render_template('reduce_lr_plateau.html', topic=topic, subtopic=subtopic, topic_id=topic_id)
            elif subtopic_id == 'neural-networks':
                return render_template('neural-networks.html', topic=topic, subtopic=subtopic, topic_id=topic_id)
            elif subtopic_id == 'optimizers':
                return render_template('optimizers.html', topic=topic, subtopic=subtopic, topic_id=topic_id)
            elif subtopic_id == 'interpolation':
                return render_template("data_preprocessing/imputation_interpolation.html", topic=topic,
                                       subtopic=subtopic,
                                       topic_id=topic_id)
            elif subtopic_id == 'ts-modeling':
                return render_template('/time_series/ts_modeling.html', topic=topic, subtopic=subtopic,
                                       topic_id=topic_id)
            elif subtopic_id == 'savgol':
                return render_template('/data_preprocessing/savgol.html', topic=topic, subtopic=subtopic,
                                       topic_id=topic_id)
            elif subtopic_id == 'tsmote':
                return render_template('/time_series/tsmote.html', topic=topic, subtopic=subtopic, topic_id=topic_id)
            elif subtopic_id == 'hybrid_model':
                return render_template('/time_series/hybrid_model.html', topic=topic, subtopic=subtopic,
                                       topic_id=topic_id)
            elif subtopic_id == 'seg-emb':
                return render_template('segmentation_embedding.html', topic=topic, subtopic=subtopic, topic_id=topic_id)
            elif subtopic_id=="reproducibility":
                return render_template("/dl/reproducibility.html",topic=topic,subtopic=subtopic,topic_id=topic_id)

            ## Stat
            elif subtopic_id == 'bias-variance':
                return render_template("/stat/bias_variance_tradeoff_note.html", topic=topic, subtopic=subtopic,
                                       topic_id=topic_id)
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
    app.run(host="0.0.0.0", debug=True)
