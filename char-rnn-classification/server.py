from bottle import route, run
from predict import *

@route('/<input_line>')
def index(input_line):
    return {'result': predict(input_line, 10)}

run(host='localhost', port=5533)
