class Tweet:
    """Class to store tweet information"""
    
    def __init__(self, tweet_id, text, polarity):
        """
        Initialize a Tweet object
        
        Parameters:
        tweet_id (str): The unique ID of the tweet
        text (str): The text content of the tweet
        polarity (float): The sentiment polarity of the tweet (-1.0 to 1.0)
        """
        self.tweet_id = tweet_id
        self.text = text
        self.polarity = polarity
    
    def get_sentiment(self):
        """
        Get the sentiment category based on polarity
        
        Returns:
        str: 'Positive', 'Negative', or 'Neutral'
        """
        if self.polarity > 0.1:
            return "Positive"
        elif self.polarity < -0.1:
            return "Negative"
        else:
            return "Neutral"
    
    def __str__(self):
        """String representation of the Tweet object"""
        return f"Tweet(id={self.tweet_id}, text='{self.text[:30]}...', polarity={self.polarity:.2f})"
