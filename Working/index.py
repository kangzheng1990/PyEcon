# -*- coding:utf-8 -*-
from bottle import route, run
@route('/')
@route('/hello/<name>')
def hello(name = u"知らない人"):
    return u"こんにちは！" + name
run(host='localhost', port=8080, debug=True)



