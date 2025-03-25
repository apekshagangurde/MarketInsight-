class Tweet:
    def __init__(self, text, processed_text, polarity, subjectivity):
        self.text = text
        self.processed_text = processed_text
        self.polarity = polarity
        self.subjectivity = subjectivity
