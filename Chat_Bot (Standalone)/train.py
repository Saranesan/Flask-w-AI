import json
import torch
import torch.nn as nn
import numpy as np
from model import NeuralNet
from torch.utils.data import Dataset, DataLoader
from util import tokenize, stem, bag_of_words

with open('intents_v3.json','r') as f:
	intents = json.load(f)["intents"][0]


class ChatDataset(Dataset):
	def __init__(self): 
		self.n_samples = len(x_train)
		self.x_data = x_train
		self.y_data = y_train

	#dataset[idx]
	def __getitem__(self,index):
		return self.x_data[index], self.y_data[index]

	def __len__(self):
		return self.n_samples


batch_size = 50
hidden_size = 5
learning_rate = 0.001
num_epoch = 1000


all_words = []
tags = []
xy = []
key = []

for k in intents.keys():
	key.append(k)


for k in key:

	for i in intents[k]:
		tag = i['tag']
		tags.append(tag)	
		

		for pat in i['patterns']:
			w = tokenize(pat)
			all_words.extend(w)
			xy.append((w,tag))

	ignore_words = ['!','.',',']
	all_words = [stem(w) for w in all_words if w not in ignore_words] 
	all_words = sorted(set(all_words))
	tags = sorted(set(tags))

	x_train = []
	y_train = []
	for (pattern_sentence,tag) in xy:
		bag = bag_of_words(pattern_sentence,all_words)
		x_train.append(bag)

		label = tags.index(tag)
		y_train.append(label)

	x_train = np.array(x_train)
	y_train = np.array(y_train)

	output_size = len(tags)
	input_size = len(all_words)

	dataset = ChatDataset()
	train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = NeuralNet(input_size,hidden_size,output_size)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	for epoch in range(num_epoch):
		for(words,labels) in train_loader:
			words = words.to(device)
			labels = labels.to(device)

			#forward
			outputs = model(words)
			loss = criterion(outputs,labels.long())

			#backwards and optimizer
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		if(epoch +1) % 100 == 0:
			print(f'epoch {epoch+1}/{num_epoch}, loss={loss.item():.4f}')
	print(f'final loss, loss={loss.item():.4f}')

	data = {
		"model_state": model.state_dict(),
		"input_size":input_size,
		"output_size":output_size,
		"hidden_size":hidden_size,
		"all_words":all_words,
		"tags":tags
	}
	FILE = f"{k}.pth"
	torch.save(data,FILE)
	print(f'training complete. file saved to {FILE}')