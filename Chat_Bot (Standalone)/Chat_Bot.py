import random
import json
import torch
from model import NeuralNet
from util import tokenize, bag_of_words


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents_v3.json','r') as f:
	intents = json.load(f)["intents"][0]

FILE = "introduction.pth"
data = torch.load(FILE)
input_size = data["input_size"]
output_size = data["output_size"]
hidden_size = data["hidden_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()



bot_name = "Interviewer"
greetings = "Hello --Player-- my name is Jon the AI Interviewer, i will be asking you a number of questions to determine your suitability \nfor the role: Junior Data Analyst at Google."



print(greetings)

while True:	
	sentence = input('You: ')
	if sentence == "quit":
		break

	sentence = tokenize(sentence)
	x = bag_of_words(sentence,all_words)
	x = x.reshape(1, x.shape[0])
	x = torch.from_numpy(x)

	output = model(x)
	_, predicted = torch.min(output,dim=1)	
	tag = tags[predicted.item()]
	print(f"output:\n{output[0]}\n")

	#print(output)
	probs = torch.softmax(output,dim=1)
	prob = probs[0][predicted.item()]


	print(f"PROBS:\n{probs}\n")
	print(f"PROB:\n{prob}\n")

	if prob.item() < 0.5:		
		for i in intents['introduction']:
			if tag == i["tag"]:
				print(f"{bot_name}:{i['responses'][0]}")			

	



# TODO 
#Change probability to only match Current Questions through the use of context-set. - Done
#Increase Pattern Count - Done
#Increasing Hidden Layers to increase accuracy. - Done
#Create final variables for names etc. - Done
#Change to Lancaster Stemmer  as Porter over stems. - Done
#Softmin instead of Softmax - Close to Done
#Evaluate idea of model per question > Model per interview. Done
#Use Lemmatization instead of stemming (remove extra char) -  Done