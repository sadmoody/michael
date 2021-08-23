import os

from flask import Flask
from flask import request, jsonify, json, render_template, abort

from .deviant.deviant import infer

from functools import wraps

def limit_content_length(max_length):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            cl = request.content_length
            if cl is not None and cl > max_length:
                abort(413)
            return f(*args, **kwargs)
        return wrapper
    return decorator

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True, static_url_path='/static')

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/api/twss', methods=['POST'])
    @limit_content_length(159)
    def twss():
        if request.method == 'POST':
            if 'sent' not in request.json:
                return jsonify(trigger=False)
            sent = request.json['sent']
            if 'sensitivity' in request.json:
                sensitivity = 17.0 - float(request.json['sensitivity'])
            else:
                sensitivity = 8.0
            if len(sent.split(' ')) > 1:
                trigger = infer(sent, sensitivity)
                print({'trigger': trigger, 'sent': sent, 'sensitivity': sensitivity})
                return jsonify(sent=sent, trigger=str(trigger))
        return jsonify(trigger=False)

    return app