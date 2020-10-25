#!/usr/bin/env python

import sys
import json
import requests_unixsocket

def do_request(path):
    session = requests_unixsocket.Session()

    request_path = path

    r = session.get('http+unix://%2Fvar%2Frun%2Fdocker.sock' + request_path)
    print(json.dumps(r.json(), indent=4))

if __name__=='__main__':
    do_request(sys.argv[1])