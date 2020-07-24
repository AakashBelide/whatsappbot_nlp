# whatsappbot_nlp
A whatsapp chatbot built using Natural Language Processing to understand the user texts and reply back with relevant message.

The chatbot is first trained using Natural Language Toolkit to understand the messages depending upon various tags and randomly choose the appropriate message to reply.
We use flask and connect it with twilio in order to receive and send messages through twilio. The chatbot is then connected to the flask app to use the chatbot on whatsapp to reply to the user.

# Python libraries used:

• nltk

• Pytorch(torch) 

• flask

• numpy

• json

• twilio

# Requirements:

• Setup a twilio account

• Download ngrok to use the localhost server as a global server and use it in the twilio sandbox settings

• You can use gunicorn to deploy the flask app online to use the trained model and then set the twilio sandbox settings
