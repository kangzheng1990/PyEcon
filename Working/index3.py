# -*- coding:utf-8 -*-
from bottle import route, run, view
@route('/<name>/<count:int>')
@view("hello_template")
def hello(name, count):
    return dict(name=name, count=count)
run(host='localhost', port=8080, debug=True)