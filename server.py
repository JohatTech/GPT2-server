import json
from flask import Flask, request, jsonify, json
from flask_cors import CORS
from chatbot import*

app = Flask(__name__)
CORS(app)

length = 25
no_repeat_ngram_size = 2
num_beams = 5
temperature = 0.7
print("Hi, let's chating")

# while(True):
#     chatbot.input_handling(input(">"))
#     chatbot.respond()

@app.route("/respond", methods =['POST', 'GET'])
def respond():
    if request.method == "POST":
        data = request.get_json()
        message = data['message']
        print(data)
        chatbot = Chatbot(length, temperature ,no_repeat_ngram_size, num_beams )
        chatbot.input_handling(message)
        prediction = chatbot.respond()
        return jsonify({'prediction':str(prediction)})        

if __name__=="__main__":
    app.run()


  
