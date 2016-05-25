import json

from flask import Flask, request
app = Flask(__name__)

@app.route('/')
def hello_world():
    data = json.loads(request.data)
    breath_data = data["breath_data"]
    print(breath_data)
    return json.dumps(data)

if __name__ == '__main__':
    app.run(debug=True)
