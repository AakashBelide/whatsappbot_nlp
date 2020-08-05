from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize



app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello, World!"

@app.route("/sms", methods=['POST'])
def sms_reply():
    with open('dataset.json', 'r') as f:
        intents = json.load(f)
    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'

    FILE = "data.pth"
    data = torch.load(FILE, map_location=map_location)
    
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data["all_words"]
    tags = data["tags"]
    model_state = data["model_state"]
    
    model = NeuralNet(input_size, hidden_size, output_size)
    model.load_state_dict(model_state)
    model.eval()
    while True:
    # Fetch the message
        msg = request.form.get('Body')
        #sentence = input(msg)
        #if sentence == "quit":
        #    break
    
        #sentence = tokenize(sentence)
        msg = tokenize(msg)
        X = bag_of_words(msg, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)
    
        output = model(X)
        _, predicted = torch.max(output, dim = 1)
        tag = tags[predicted.item()]
    
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
    
        if prob.item() > 0.75:
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    # Create reply
                    resp = MessagingResponse()
                    resp.message(random.choice(intent['responses']).format(msg))
            
        else:
            resp = MessagingResponse()
            resp.message("I do not understand...".format(msg))
        """Respond to incoming calls with a simple text message."""
    
        return str(resp)

if __name__ == "__main__":
    app.run(debug=False)