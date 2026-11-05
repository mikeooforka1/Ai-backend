import sys,json
print(json.dumps({'event':'ready','capabilities':{}}),flush=True)
for line in sys.stdin:
    message=json.loads(line)
    print(json.dumps({'id':message['id'],'result':{'command':message['command']}}),flush=True)
