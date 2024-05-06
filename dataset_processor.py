# Processes Old Instructions Dataset (we were going to use)

import json


def process_json_file(file_path):
    # Read the input JSON file
    with open(file_path, "r") as file:
        data = json.load(file)

    # Iterate through each JSON object and extract the "instruction" elements
    instructions = []
    for obj in data:
        if "instruction" in obj:
            instructions.append(obj["instruction"])

    # Create a new JSON file with the extracted instructions
    output_data = json.dumps(instructions)
    with open("queries.json", "w") as file:
        file.write(output_data)


# Call the function with the path to your JSON file
process_json_file("seq.json")
