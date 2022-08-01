import random
import json
import torch
from Chat_Bot.model import NeuralNet
from Chat_Bot.util import tokenize, bag_of_words

def interview_response(sentence):

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	with open('Chat_Bot/intents_v3.json','r') as f:
		intents = json.load(f)["intents"][0]

	FILE = "Chat_Bot/introduction.pth"
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

	


	sentence = tokenize(sentence)
	x = bag_of_words(sentence,all_words)
	x = x.reshape(1, x.shape[0])
	x = torch.from_numpy(x)

	output = model(x)
	_, predicted = torch.min(output,dim=1)	
	tag = tags[predicted.item()]
	print(f"output:\n{output[0]}\n") #for testing only! 
	
	probs = torch.softmax(output,dim=1)
	prob = probs[0][predicted.item()]


	print(f"PROBS:\n{probs}\n")#for testing only! 
	print(f"PROB:\n{prob}\n")#for testing only! 

	if prob.item() < 0.5:		
		for i in intents['introduction']:
			if tag == i["tag"]:
				return i['responses'][0]		

		


