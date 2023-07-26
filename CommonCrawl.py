import transformers
import os
import datetime
import numpy as np

global user_input
user_input = ""

class ChatBot():
    def __init__(self, name):
        print("----- Starting up", name, "-----")
        self.name = name

    def input_to_text(self):
        user_input = input("Me  --> ")

    @staticmethod
    def text_to_response(text):
        print("XLM-V (Common Crawl) --> ", text)
        
    def wake_up(self, text):
        return True if self.name in str(user_input).lower() else False

    @staticmethod
    def action_time():
        return datetime.datetime.now().time().strftime('%H:%M')

if __name__ == "__main__":
    
    ai = ChatBot(name="XLM-V")
    nlp = transformers.pipeline("conversational", model="facebook/xlm-v-base")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    ex=True
    while ex:
        ai.input_to_text()

        if ai.wake_up(str(user_input)) is True:
            res = "Hello I am XLM-V."
        elif "time" in str(user_input):
            res = ai.action_time()
        elif any(i in str(user_input) for i in ["thank","thanks"]):
            res = np.random.choice(["you're welcome!","anytime!","no problem!","cool!","I'm here if you need me!","mention not"])
        elif any(i in str(user_input) for i in ["exit","close"]):
            res = np.random.choice(["Tata","Have a good day","Bye","Goodbye","Hope to meet soon","peace out!"])
            ex=False
        else:   
            if str(user_input)=="ERROR":
                res="Sorry, come again?"
            else:
                chat = nlp(transformers.Conversation(str(user_input)), pad_token_id=50256)
                res = str(chat)
                res = res[res.find("bot >> ")+6:].strip()

        ai.text_to_response(res)
    print("----- Closing down XLM-V (Common Crawl) -----")