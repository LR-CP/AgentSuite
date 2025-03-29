# AgentSuite

AgentSuite is a versatile agent-based system that enables interaction with different types of AI agents through a streamlined web interface. The suite includes document summarization, chat capabilities, and codespace analysis features.

## Features

- **Multiple Agent Types**:
  - Chat Agent: Interactive conversations with LLM models
  - Document Summarization Agent: Summarize various document formats
  - Codespace Guru: Analyze and understand code repositories
- **Conversation Management**: Save and load chat conversations
- **Document Support**: Handle PDF, DOCX, CSV, and text files
- **Local LLM Integration**: Uses Ollama for local model inference
- **Web Interface**: Built with Streamlit for easy interaction

## Prerequisites

- Python 3.x
- [Ollama](https://github.com/jmorganca/ollama) installed and running locally
- Git (for cloning the repository)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/LR-CP/AgentSuite.git
cd AgentSuite
```

2. Run the setup script to create a virtual environment and install dependencies:
```bash
python setup.py
```

3. Activate the virtual environment:

On Windows:
```bash
.\venv\Scripts\activate
```

On Unix/MacOS:
```bash
source venv/bin/activate
```

## Configuration

1. Ensure Ollama is running locally on `http://localhost:11434` (or another IP if being hosted externally)
2. Default model is set to `llama3.2` - you can change this in the UI

## Running the Application

1. Make sure your virtual environment is activated
2. Start the Streamlit app:
```bash
streamlit run main.py
```
3. Open your browser and navigate to `http://localhost:8501`

## Usage

### Chat Agent
- Select "ChatAgent" from the sidebar
- Type messages in the chat input
- Save conversations with custom names
- Load previous conversations

### Document Summarization
- Select "SummarizeDocAgent" from the sidebar
- Upload supported documents (PDF, DOCX, CSV, TXT)
- Get AI-generated summaries

### Codespace Guru
- Select "CodespaceGuru" from the sidebar
- Choose a code repository folder
- Get insights and analysis about the codebase
- Ask specific questions about the code

## Project Structure

```
AgentSuite/
├── agents/           # Agent implementations
├── conversations/    # Saved chat conversations
├── core/            # Core functionality
├── factory/         # Agent creation and storage
├── tools/           # Tool implementations
├── ui/              # Streamlit interface
└── vector_db/       # Vector database storage
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Support

For support, please open an issue in the GitHub repository.