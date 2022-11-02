import json


def save_json_file (json_compatible_data, name):
    with open(name +  '.json', 'w') as fp:
        text = json.dumps(json_compatible_data)
        fp.write(text)
