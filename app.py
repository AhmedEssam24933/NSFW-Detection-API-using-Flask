import os
from flask import Flask, request, jsonify
from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO
import logging

# Initialize Flask app
app = Flask(__name__)

# Set the path to your model directory and upload folder
model_dir = r'NSFW'
upload_folder = './uploads'
os.makedirs(upload_folder, exist_ok=True)  # Create the upload folder if it doesn't exist

# Initialize the NSFW detection pipeline
nsfw_detector = pipeline("image-classification", model=model_dir)

# Configure external API details
external_api_url = "http://localhost:8000/post/images"
external_api_headers = {
    'Content-Type': 'application/json',
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MTgzNywicm9sZSI6IlNUVURFTlQiLCJpYXQiOjE3MzMzMTExMzEsImV4cCI6MTczNDUyMDczMX0.8FOdsBZLvnwcFkpWSU3uo0gWDGhKaNSJcI3nYahCE7c"
}

# Function to process a single image from a link
def process_image_from_link(link):
    try:
        # Download the image from the link
        response = requests.get(link)
        response.raise_for_status()  # Raise an error for bad responses

        # Open the image directly from the response content
        image_pil = Image.open(BytesIO(response.content))

        # Run the model on the image (no resizing or RGB conversion)
        results = nsfw_detector(image_pil)

        # Extract the score of the result (assuming the NSFW label is the one we're checking)
        nsfw_score = next((result['score'] for result in results if result['label'] == 'nsfw'), 0)

        # Return True if the score exceeds the threshold
        return nsfw_score > 0.5
    except requests.exceptions.RequestException as e:
        # Handle network errors
        logging.error(f"Network error while processing image from link ({link}): {str(e)}")
        return False
    except Exception as e:
        # Log any other errors
        logging.error(f"Error processing image from link ({link}): {str(e)}")
        return False

# Route to process multiple image links
@app.route('/process-images', methods=['POST'])
def process_images():
    # Get the JSON payload from the request
    data = request.json
    if not data or 'images_links' not in data or 'post_id' not in data:
        return jsonify({"error": "Invalid request. 'image_links' and 'post_id' are required"}), 400

    # Extract the list of image links and the post ID
    image_links = data['images_links']
    post_id = data['post_id']
    if not isinstance(image_links, list):
        return jsonify({"error": "Image links should be provided as a list"}), 400

    # Process each image link sequentially
    for link in image_links:
        result = process_image_from_link(link)
        if result:  # If any image is classified as NSFW, send DELETE request
            try:
                delete_response = requests.delete(
                    external_api_url,
                    json={"postId": post_id},
                    headers=external_api_headers
                )
                if delete_response.status_code == 200:
                    logging.info(f"Post {post_id} was successfully deleted due to NSFW content.")
                else:
                    logging.error(f"Failed to delete post {post_id}: {delete_response.status_code}")
                    logging.error("Response: %s", delete_response.text)
            except requests.RequestException as e:
                logging.error(f"Error sending request to external API: {e}")
            return jsonify({"postId": post_id}), 200

    # If none of the images are NSFW, return False
    return '', 204

# Run the app
if __name__ == '__main__':
    app.run(debug=True)















