import json

from flask import Flask, request
app = Flask(__name__)

@app.route('/')
def hello_world():
    print(type(request.data))
    print(request.data)
    data = json.loads(request.data)
    print(type(data))
    breath_data = data["breath_data"]
    print(breath_data)
    return json.dumps(data)

if __name__ == '__main__':
    app.run(debug=True)
