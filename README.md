# Knowledge-Based-Agent

## Overview
This project leverages OpenAI's powerful models, LangChain's agent capabilities, and Streamlit to create a responsive and intelligent application. It connects to a PostgreSQL database, handles errors gracefully, and utilizes advanced AI techniques to process and respond to user inputs dynamically.

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
set OPENAI_API_KEY: "Your OpenAI API key"
```
Database credentials in the environment or configuration file.

## Usage

Start the Streamlit application:
```bash
streamlit run ui.py
```
Navigate to http://localhost:8501 in your browser to view the application.

## File Structure
- **data_preprocessing.py**: Handles data loading and database interactions.
- **inference_engine.py**: Contains the core AI and decision-making capabilities using LangChain agents.
- **ui.py**: Manages the Streamlit frontend interface.


## Contributing
Contributions are welcome! Please fork the repository and open a pull request with your improvements.

## License
Distributed under the MIT License. See LICENSE for more information.


