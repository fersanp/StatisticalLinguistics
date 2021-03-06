import json

with open('/tmp/foo', 'r') as f:
        json_data = json.load(f)

d = json_data["data"]
for art in d:
        id = art["id"]
        if "Materials" in id:
                print(id)
                print("--------------------------------")
#json_data = json.loads("/tmp/art.json")
#print(json_data)


