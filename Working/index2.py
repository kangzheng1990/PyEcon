# -*- coding:utf-8 -*-
from bottle import route, run
@route('/<name:int>')
def hello(name):
    return str(name)
run(host='localhost', port=8080, debug=True)