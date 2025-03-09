import base64
import os
from threading import Lock, Thread



from transformers import AutoTokenizer, AutoModelForCausalLM

import cv2
import openai
from cv2 import VideoCapture, imencode
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from pyaudio import PyAudio, paInt16
from speech_recognition import Microphone, Recognizer, UnknownValueError

# Load environment variables
load_dotenv()


class WebcamStream:
    def __init__(self):
        self.stream = VideoCapture(index=0)
        _, self.frame = self.stream.read()
        self.running = False
        self.lock = Lock()

    def start(self):
        if self.running:
            return self

        self.running = True

        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.running:
            _, frame = self.stream.read()

            with self.lock:
                self.frame = frame

    def read(self, encode=False):
        with self.lock:
            frame = self.frame.copy()

        if encode:
            _, buffer = imencode(".jpeg", frame)
            return base64.b64encode(buffer)

        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stream.release()


class Assistant:
    def __init__(self, model):
        self.chain = self._create_inference_chain(model)
        self.current_player = None
        self.is_speaking = False

    def stop_speaking(self):
        """Stop the current audio playback if any."""
        if self.current_player and self.is_speaking:
            self.is_speaking = False
            self.current_player.stop_stream()
            self.current_player.close()
            self.current_player = None

    def answer(self, prompt, image):
        if not prompt:
            return

        print("Prompt:", prompt)
        
        # Buffer to accumulate the full response
        full_response = []
        
        # Stream the response
        for chunk in self.chain.stream(
            {"prompt": prompt, "image_base64": image.decode()},
            config={"configurable": {"session_id": "unused"}},
        ):
            if chunk:  # Skip empty chunks
                print("Chunk:", chunk)
                full_response.append(chunk)
        
        # Combine all chunks into final response
        response = "".join(full_response).strip()
        print("Full Response:", response)

        if response:
            self._text_to_speech(response)

    def _text_to_speech(self, response):
        # Stop any existing playback
        self.stop_speaking()
        
        self.current_player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)
        self.is_speaking = True

        try:
            with openai.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="alloy",
                response_format="pcm",
                input=response,
            ) as stream:
                for chunk in stream.iter_bytes(chunk_size=1024):
                    if not self.is_speaking:
                        break
                    self.current_player.write(chunk)
        finally:
            self.stop_speaking()

    def _create_inference_chain(self, model):
        SYSTEM_PROMPT = """
        You are a witty assistant that will use the chat history and the image 
        provided by the user to answer its questions.

        Use few words in your answers. Go straight to the point. Do not use any
        emoticons or emojis. Do not ask the user any questions.

        Be friendly and helpful. Show some personality. Do not be too formal.
        If you don't understand a question, ask the user to explain it to avoid hallucination.
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64,{image_base64}",
                        },
                    ],
                ),
            ]
        )

        chain = prompt_template | model | StrOutputParser()

        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )


# Retrieve OpenAI API key from .env file
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the webcam stream
webcam_stream = WebcamStream().start()

# Initialize the model with OpenAI
model = ChatOpenAI(
    api_key=openai_api_key,
    model="gpt-4-turbo",
    max_tokens=1024,
    temperature=0.7,
    streaming=True
)

# Initialize assistant
assistant = Assistant(model)


def audio_callback(recognizer, audio):
    try:
        prompt = recognizer.recognize_whisper(audio, model="base", language="english")
        assistant.answer(prompt, webcam_stream.read(encode=True))

    except UnknownValueError:
        print("There was an error processing the audio.")


# Speech recognition setup
recognizer = Recognizer()
microphone = Microphone()

with microphone as source:
    recognizer.adjust_for_ambient_noise(source)

stop_listening = recognizer.listen_in_background(microphone, audio_callback)

# Main loop to display webcam feed
while True:
    cv2.imshow("webcam", webcam_stream.read())
    if cv2.waitKey(1) in [27, ord("q")]:
        break

# Cleanup
webcam_stream.stop()
cv2.destroyAllWindows()
stop_listening(wait_for_stop=False)