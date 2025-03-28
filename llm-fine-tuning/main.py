from google import genai
from dotenv import load_dotenv
import os
import sys
import time

# Import the chatbot model
from chatbot import TherapyChatbot

# Load environment variables
load_dotenv()

def type_out(text, delay=0.01):
    """
    Simulate typing out text character by character
    :param text: Text to be typed out
    :param delay: Delay between characters (default 0.05 seconds)
    """
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()  # New line after typing

def main():
    # Initialize Google AI model
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    chat_history = []
    
    # Create therapy chatbot instance
    therapy_bot = TherapyChatbot(client)
    
    # Conversation loop
    print("Therapy Chatbot: Hello! I'm here to listen and support you.")
    
    while True:
        user_input = input("You: ")
        
        # Exit condition
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Therapy Chatbot: Take care. Remember, you're not alone.")
            break
        
        # Process message
        result = therapy_bot.process_message(user_input, chat_history)

        chat_history = result['chat_history']
        
        # Display response and emotion context
        print("\nChatbot Response:")
        type_out(result['response'].text)
        print("\nEmotion Insights:")
        type_out(result['emotion_context']['emotion_summary'])

if __name__ == "__main__":
    main()