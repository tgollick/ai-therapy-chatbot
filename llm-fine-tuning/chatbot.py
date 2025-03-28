from sentiment_analyzer import SemanticEmotionAnalyzer
from response_generation import ResponseGenerator

class TherapyChatbot:
    def __init__(self, client):
        """
        Initialize the therapy chatbot
        
        :param chat_model: Initialized chat model (e.g., Gemini)
        """
        self.emotion_detector = SemanticEmotionAnalyzer()
        self.response_generator = ResponseGenerator(client)
    
    def process_message(self, user_input, chat_history):
        """
        Process user message with emotion analysis and response generation
        
        :param user_input: User's input message
        :return: Dictionary with response and emotion context
        """
        # Detect emotional context
        emotion_analysis = self.emotion_detector.detect_emotion(user_input)
        
        # Add emotion summary
        emotion_analysis['emotion_summary'] = self.emotion_detector.generate_emotion_summary(emotion_analysis)
        
        # Generate response
        response = self.response_generator.generate_response(
            user_input, 
            emotion_analysis,
            chat_history
        )

        chat_history.append({
            'user_input': user_input,
            'emotion_analysis': emotion_analysis
        })

        
        return {
            'response': response,
            'emotion_context': emotion_analysis,
            'chat_history': chat_history
        }