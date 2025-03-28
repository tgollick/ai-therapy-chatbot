class ResponseGenerator:
    def __init__(self, client):
        """
        Initialize response generator with a chat model
        
        :param chat_model: Initialized chat model (e.g., Gemini)
        """
        self.client = client

    def generate_emotion_aware_prompt(self, user_input, emotion_analysis, chat_history):
        """
        Create a context-rich prompt incorporating emotional insights
        
        :param user_input: Original user message
        :param emotion_analysis: Emotion analysis dictionary
        :return: Enhanced prompt for response generation
        """
        emotion_context = f"""
        CHAT HISTORY: {chat_history}

        EMOTIONAL CONTEXT:
        - Dominant Emotion: {emotion_analysis['dominant_emotion']}
        - Sentiment Intensity: {emotion_analysis['sentiment']['compound']}
        
        USER INPUT:
        {user_input}
        
        RESPONSE GUIDELINES:
        - Provide empathetic, supportive response
        - Acknowledge the detected emotional state
        - Offer constructive, supportive guidance
        """
        
        return emotion_context
    
    def generate_response(self, user_input, emotion_analysis, chat_history):
        """
        Generate a response based on user input and emotional context
        
        :param user_input: Original user message
        :param emotion_analysis: Emotion analysis dictionary
        :return: Generated response
        """
        # Create emotion-aware prompt
        enhanced_prompt = self.generate_emotion_aware_prompt(
            user_input, 
            emotion_analysis,
            chat_history
        )
        
        # Generate response using chat model
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=enhanced_prompt,
        )
        
        return response