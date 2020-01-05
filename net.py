import numpy as numpy
import torch
import time


class Net(nn.Module):

	def __init__(self, lrate, loss_fn, in_size, out_size):
		
		# Initialize variables
		self.loss_fn = loss_fn

		# Initialize (single) device
		if torch.cuda.is_available():
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cpu')

		# Initialize model
		conv1 = nn.Sequential(nn.Conv2d(3, 18, 5), nn.LeakyRelU(), nn.MaxPool2d(2))
		conv2 = nn.Sequential(nn.Conv2d(18, 24, 2), nn.LeakyRelU(), nn.MaxPool2d(2))
		l1 = nn.Sequential(nn.Linear(in_size, ), nn.LeakyRelU())
		l2 = nn.Sequential(nn.Linear(, ), nn.LeakyRelU())
		l3 = nn.Sequential(nn.Linear(, out_size), nn.LeakyRelU())
		self.model = nn.Sequential(conv1, conv2, l1, l2, l3).to(device=self.device)

		# Initialize optimizer
		self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, weight_decay=1)


	def forward(self, images):

		# Images should already be in friendly format
		return self.model(image.to(device=self.device))


	def step(self, images, results):

		self.optimizer.sero_grad()
		outputs = self.forward(image)

		loss = self.loss_fn(outputs, results.to(device=self.device()))

		loss.backward()
		self.optimizer.step()
		return loss.item()


def get_number(outputs):

	total = outputs.sum()
	answer = 0

	for idx in range(len(outputs)):
		answer += idx * outputs[idx] / total


def fit(train_set, train_labels, dev_set, n_iter, batch_size=100):

	lr = 0.00005
	total_loss = [0] * n_iter
	results = 
	net = Net(lr, nn.CrossEntropyLoss(), train_set.size()[1], 11)
	start = time.time()

	for epoch in range(n_iter):
		print("Starting epoch: %d" % epoch)

		loss = 0
		train_loader = torch.utils.data.DataLoader(utils.data.TensorDataset(train_set, train_labels), batch_size=batch_size, shuffle=True)

		for batch, labels in train_loader:
			loss += net.step(batch, labels)
		total_loss[epoch] = loss
		print("Done with epoch: %d at time %d seconds" % (epoch, time.time() - start))

	# Pass the dev set through the model, and get actual numbers from the outputs
	outputs = net.eval().forward(dev_set)
	for i in range(len(outputs)):
		total = outputs[i].sum()
		results[i] = 0

		for idx in range(len(outputs[i])):
			results[i] += idx * outputs[i][idx] / total

	# Finally, return the loss, results, and the net
	return total_loss, results, net.eval()
