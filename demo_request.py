
import requests
import json
import numpy as np
import argparse

def get_sample_data(model_type):
    """
    Generates sample data based on the model type.
    """
    if model_type == 'cnn':
        # Shape (1, 3, 224, 224) for a single image with 3 channels and 224x224 resolution
        return np.random.rand(1, 3, 224, 224).tolist()
    elif model_type == 'llm':
        # Shape (1, 128) for a single sequence of 128 tokens
        return np.random.randint(0, 1000, (1, 128)).tolist()
    elif model_type == 'multimodal':
        # For multimodal, the input is more complex.
        # This is a placeholder and should be adapted to the actual model's needs.
        return np.random.rand(1, 3, 224, 224).tolist()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def main():
    parser = argparse.ArgumentParser(description='Send a demo inference request to the API service.')
    parser.add_argument('model_type', type=str, default='cnn', nargs='?',
                        choices=['cnn', 'llm', 'multimodal'],
                        help='The type of model to send the request to (default: cnn)')
    args = parser.parse_args()

    # URL of the API endpoint
    url = f"http://127.0.0.1:3434/infer/{args.model_type}"

    # Get sample data for the specified model type
    try:
        sample_data = get_sample_data(args.model_type)
    except ValueError as e:
        print(e)
        return

    # Create the JSON payload
    payload = {"data": sample_data}

    try:
        # Send the POST request
        print(f"Sending request to {url} with data for model type '{args.model_type}'...")
        response = requests.post(url, json=payload)

        # Check if the request was successful
        if response.status_code == 200:
            print("Request successful!")
            print("Response:")
            print(response.json())
        else:
            print(f"Request failed with status code: {response.status_code}")
            print("Response:")
            print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
