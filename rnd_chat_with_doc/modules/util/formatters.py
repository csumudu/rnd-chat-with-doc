import json

def get_formatted_response(content):
    data = {"type": "message","content" : content}
    json_data = json.dumps(data)  # Convert dictionary to JSON string
    return json_data