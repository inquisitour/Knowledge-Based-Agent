# Knowledge-Based-Agent

## Overview
Knowledge based agent is powered by sophisticated integrations with OpenAI, LangChain, and Streamlit, coupled with a robust PostgreSQL database backend. Our focus has been on creating an agent that not only handles tasks autonomously but also adapts and responds to user input with high intelligence and contextual awareness. By seamlessly merging these technologies, we ensure that our agent is not only responsive and dynamic but also capable of learning and evolving over time to better serve user needs.
## Features
- **Database Integration**: Connects to PostgreSQL database to manage and retrieve data efficiently.
- **OpenAI Integration**: Utilizes OpenAI's APIs for embeddings and chat functionalities to generate context-aware responses.
- **LangChain Integration**: Implements LangChain agents to handle decision-making processes, enhancing the application's ability to respond intelligently based on user input and stored data.
- **Streamlit Application**: Provides a user-friendly web interface for interaction, allowing users to input questions and receive answers dynamically.
- **Error Handling**: Features robust error handling for database interactions and API calls, ensuring the application runs smoothly under various conditions.
  
## Technology Stack
- Python
- OpenAI API
- PostgreSQL
- Streamlit
- LangChain
- AWS

## Installation

### Prerequisites
- Python 3.8 or later
- PostgreSQL server
- OpenAI API key

### Clone the repository:
```bash
git clone https://github.com/inquisitour/Knowledge-Based-Agent.git
```

### Install the required packages:
```bash
pip install -r requirements.txt
```
### Set up environment variables:
```bash
set OPENAI_API_KEY "Your OpenAI API key"
```
Database credentials in the environment or configuration file.

## Usage
### Local Setup
Setup your database environment and navigate to the local folder:
```bash
cd 'Local Agent'
```
Start the Streamlit application:
```bash
streamlit run local_interface.py
```
Navigate to http://localhost:8501 in your browser to view the application.

### AWS Lambda Setup
Setup your database environment and navigate to the lambda folder:
```bash
cd 'Cloud Agent'
```
Review the contents of the lambda folder:
```bash
ls
```
Open the lambda code files in a text editor:
```bash
code .
```
Check the status of your Lambda function using the AWS CLI:
```bash
aws lambda list-functions
```
Deploy your Lambda function using the AWS CLI (if you have the AWS CLI configured):
```bash
sls deploy
```
Deploy your Lambda function using the AWS Lambda Console or the AWS CLI.
Set the following environment variables in your Lambda function configuration:
- **OPENAI_API_KEY**: Your OpenAI API key.
- **S3_BUCKET_NAME**: The name of your S3 bucket containing the data file.
- **S3_FILE_KEY**: The key (path) of your data file in the S3 bucket.

## File Structure
- **Architecture**: Contains an architecture diagram for the current workflow.
- **Cloud Agent**: Contains the AWS Lambda version of the code.
- **Local Agent**: Contains the local version of the code.
- **.gitignore**: Specifies intentionally untracked files to ignore.
- **README.md**: This file.
- **requirements.txt**: Lists the required Python packages for installation.


## Contributing
Contributions are welcome! Please fork the repository and open a pull request with your improvements.

## License
Distributed under the MIT License. See LICENSE for more information.


