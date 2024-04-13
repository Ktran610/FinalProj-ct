# Duy Tan University Admission Chatbot

This repository contains the source code for a chatbot designed to assist with admissions queries at Duy Tan University. The chatbot leverages LangChain to provide accurate and helpful responses to prospective students.

## Prerequisites

Before you begin, ensure you have Python 3.9 or higher installed on your system. You can download Python from the official Python website.

## Installation

1. **Clone the Repository**:
   - Clone this repository to your local machine.
   - Change directory to the cloned repository.
   ```bash
   git clone https://github.com/Ktran610/FinalProj-ct/tree/api
   cd FinalProj-ct
   ```

2. **Set up the OPENAI_API_KEY Environment Variable**:
   - Before running the chatbot, you must provide your OpenAI API key. For security reasons, it is recommended to set this as an environment variable.
   - Instructions for setting the environment variable are included depending on your operating system. Please ensure you replace `"Your_OpenAI_API_Key"` with your actual OpenAI API key and restart your terminal or command prompt to ensure the environment variable is loaded.

3. **Install Dependencies**:
   - Install the necessary libraries and dependencies listed in the `requirements.txt` file.
   ```
   pip install -r requirements.txt
   ```
## Running the Chatbot

To start the chatbot API, execute the `fast_api.py` file within the `src` directory.
```
python src/fast_api.py
```
## Testing the Chatbot

After launching the API, you can test the chatbot by accessing the FastAPI documentation at `localhost:5566` using your web browser. This will open the FastAPI documentation where you can interact with the chatbot by sending requests through the provided interface.

## Contributions

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have suggestions or need any assistance.

Thank you for using or contributing to the Duy Tan University Admission Chatbot!
