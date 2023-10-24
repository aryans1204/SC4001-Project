import torch
from data import celeba_train_dataloader


def get_inception():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    return model


class MyInception():
    def __init__(self):
        model = get_inception()


def train_one_step(model, loss, optimizer, dataloader):
    lf = 0
    for _, (X, y) in enumerate(dataloader):
        preds = model(X)
        l = loss(preds, y)
        lf += l.item()
        l.backward()
        optimizer.step()
    
    return lf/len(dataloader.size)

def trainer(dataloader, loss, optimizer, epochs, weights=None, save_dir=None, model=None):
    if model is None:
        model = get_inception()
    losses = []
    if weights:
        model.load_state_dict(torch.load(weights))
    
    for epoch in epochs:
        optimizer.zero_grad()
        lf = 0
        for _, (X, y) in enumerate(dataloader):
            preds = model(X)
            l = loss(preds, y)
            lf += l.item()
            l.backward()
            optimizer.step()
        
        losses.append(lf/len(dataloader.size))
        print(f"Loss at epoch {epoch+1} is {losses[-1]}")

    if save_dir:
        torch.save(model.state_dict, save_dir)
    
    return losses

def main(args):
    


        
    

