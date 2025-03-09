# Alloy Voice Assistant

A modern voice assistant with both speech and text chat capabilities, built with Next.js and Python.

## Features

- Voice input enabled by default
- Text chat support
- Real-time speech-to-text conversion
- Modern, responsive UI
- Seamless integration with OpenAI's GPT-4

## Prerequisites

- Node.js 18+ and npm
- Python 3.8+
- OpenAI API key

## Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd alloy-voice-assistant
```

2. Set up the frontend:

```bash
cd frontend
npm install
```

3. Set up the backend:

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

## Running the Application

1. Start the backend server:

```bash
cd backend
python app.py
```

2. In a new terminal, start the frontend development server:

```bash
cd frontend
npm run dev
```

3. Open your browser and navigate to `http://localhost:3000`

## Usage

- The voice assistant starts listening automatically when you open the page
- Click the microphone button to toggle voice input
- Type in the text input field for text-based chat
- Press Enter or click the send button to send text messages

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
