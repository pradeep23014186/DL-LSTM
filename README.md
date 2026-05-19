# DL- Developing a Deep Learning Model for NER using LSTM

## AIM
To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
Traditional machine learning models and basic Recurrent Neural Networks (RNNs) often struggle to effectively learn long-term patterns in sequential data due to limitations such as the vanishing gradient problem. This makes accurate prediction and analysis of time-series data challenging, especially when future outcomes depend on information from much earlier time steps.

The problem addressed in this project is to develop and implement a Long Short-Term Memory (LSTM) based deep learning model capable of capturing both short-term and long-term dependencies in sequential data. The objective is to preprocess the dataset, train the LSTM model, and evaluate its ability to predict future values with higher accuracy and stability.

This project demonstrates how LSTM networks, an advanced form of recurrent neural networks, can improve sequence modeling and forecasting performance in applications such as stock price prediction, weather forecasting, and trend analysis.

### DATASET:
![alt text](Output-img/Dataset.png)

## DESIGN STEPS
### STEP 1: 

Load and prepare data

### STEP 2: 

Prepare a Dataset class

### STEP 3: 

Define the Model

### STEP 4: 

Write Training and Evaluation Functions

### STEP 5: 

Run training and evaluation

### STEP 6: 

Plot the loss




## PROGRAM

### Name: Krishna Prasad S

### Register Number: 212223230108

```python
class BiLSTMTagger(nn.Module):
  def __init__(self, vocab_size, tagset_size, embedding_dim=50, hidden_dim=100):
    super(BiLSTMTagger, self) .__init__()
    self.embedding=nn.Embedding(vocab_size,embedding_dim)
    self.dropout=nn.Dropout(0.1)
    self.lstm=nn.LSTM(embedding_dim,hidden_dim,batch_first=True,bidirectional=True)
    self.fc=nn.Linear(hidden_dim*2,tagset_size)
  def forward(self,x):
    x=self.embedding(x)
    x=self.dropout(x)
    x, _= self.lstm(x)
    return self.fc(x)

        


model=BiLSTMTagger(len(word2idx)+1,len(tag2idx)).to(device)
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)


# Training and Evaluation Functions
# Training and Evaluation Functions
def train_model(model,train_loader,test_loader,loss_fn,optimixer,epochs=3):
  train_losses,val_losses=[], []
  for epoch in range(epochs):
    model.train()
    total_loss=0
    for batch in train_loader:
      input_ids=batch["input_ids"].to(device)
      labels=batch["labels"].to(device)
      optimizer.zero_grad()
      outputs=model(input_ids)
      loss=loss_fn(outputs.view(-1,len(tag2idx)),labels.view(-1))
      loss.backward()
      optimizer.step()
      total_loss+=loss.item()
    train_losses.append(total_loss)

    model.eval()
    val_loss=0
    with torch.no_grad():
      for batch in test_loader:
        input_ids=batch["input_ids"].to(device)
        labels=batch["labels"].to(device)
        outputs=model(input_ids)
        loss=loss_fn(outputs.view(-1,len(tag2idx)), labels.view(-1))
        val_loss+=loss.item()
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}: Train Loss = {total_loss :.4f}, Val Loss = {val_loss :.4f}")

  return train_losses, val_losses

```

### OUTPUT

## Loss Vs Epoch Plot

![alt text](Output-img/lossvsepoch.png)

### Sample Text Prediction

![alt text](Output-img/output.png)

## RESULT
Thus, a Long Short-Term Memory (LSTM) model for predicting stock prices using historical closing price data has been developed successfully.
