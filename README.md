
# Brazil Chatbot

Brazil Chatbot is a Flask-based chatbot application designed to utilize the capabilities of OpenAI and Pinecone, providing a sophisticated conversational experience. This application is developed to help users interact with an intelligent chatbot system easily and effectively.

## Getting Started

These instructions will guide you through setting up and running the Brazil Chatbot on your local machine for development and testing purposes.

### Prerequisites

Ensure you have Conda installed on your system. If not, you can install it by following the instructions on the [official Conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

### Installation

Follow these steps to set up your development environment:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/AyushSengar27/chatrepo
   cd chatrepo
   ```

2. **Create and Activate the Conda Environment**
   ```bash
   conda create -n chatbot python=3.8 -y
   conda activate chatbot
   ```

3. **Install Required Packages**
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

Set up the necessary environment variables in the `.env` file within the project directory as follows:

```plaintext
PINECONE_API_KEY=Your_Pinecone_API_key_here
PINECONE_INDEX=Your_Pinecone_index_name_here
INDEX_DIMENSIONS=Dimensions_for_your_Pinecone_index_here
OPENAI_API_KEY=Your_OpenAI_API_key_here
```

### Running the Application

Execute these commands to start the application:

```bash
python store_index.py  # Initialize the Pinecone index
python app.py          # Launch the Flask application
```

Navigate to `http://127.0.0.1:8080` to interact with the Brazil Chatbot.

### Testing

Run the following command to execute tests:

```bash
pytest tests.py
```

## Built With

* [Flask](http://flask.pocoo.org/) - The web framework used
* [Pinecone](https://www.pinecone.io/) - Vector database for vector search applications
* [OpenAI](https://www.openai.com/) - AI research and deployment company

## Author

* **Ayush Sengar**
  * Email: [ayushsengar27@gmail.com](mailto:ayushsengar27@gmail.com)
  * GitHub: [AyushSengar27](https://github.com/AyushSengar27)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Acknowledgments

* Thanks to everyone who has contributed to the development of this application.

