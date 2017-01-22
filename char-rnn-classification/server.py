from bottle import route, run
from predict import *

@route('/<input_name>')
def index(input_name):
    return {'result': predict(input_name, 10)}

run(host='localhost', port=5533)
