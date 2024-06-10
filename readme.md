# AI Interview Trainer

Welcome to the AI Interview Trainer project! This server-side application analyzes interview responses to help candidates improve their interview skills by providing feedback on audio, video, and text responses.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The AI Interview Trainer takes in a resume and job description, generates interview questions, and analyzes candidate responses. This project aims to enhance the interview preparation process by providing detailed feedback on audio, video, and text inputs.

## Features
- **Resume and Job Description Parsing:** Generate relevant interview questions.
- **Audio Analysis:** Evaluate voice quality using AssemblyAI.
- **Video Analysis:** Analyze body language and expressions using OpenCV and custom CNN models.
- **Text Analysis:** Detect filler words and overall response quality.
- **Relevance Check:** Compare candidate responses to AI-generated expected answers.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/STSonyThomas/API_FP_v1.git
   cd API_FP_v1
   ```
2. Set up Virtual Environment:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```
3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```   
## Usage
1. Run the server:
    ```sh
    flask run
    ```
2. Use Tools like Postman or curl to interact with the API:

## API Endpoints
* GET /questions: Retrieve generated interview questions.
* POST /analyze/audio: Submit audio for analysis.
* POST /analyze/video: Submit video for analysis.
* POST /analyze/text: Submit text responses for analysis.

## Technologies Used
* Flask: For the server-side logic.
* AssemblyAI: For audio analysis.
* OpenCV & CNN: For video analysis.
* Natural Language Processing: For text analysis.
* FFmpeg: For audio extraction.

## Contributing
We welcome contributions! Please fork the repository and create a pull request with your changes. Ensure your code follows the project's coding standards and is well-documented.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.txt) file for more details.

