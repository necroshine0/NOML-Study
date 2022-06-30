from IPython.display import display, clear_output

def train(model, optimizer, criterion, train_loader, valid_loader, test_loader, epochs=10, DEVICE='cpu'):
    
    metrics_dict = {
        "Epoch": [],
        "Train RMSE": [],
        "Valid RMSE": [],
        "Test RMSE": [],
    }
    for epoch in range(epochs):
        metrics_dict["Epoch"].append(epoch + 1)

        loss_at_train = 0 
        for batch in train_loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            y_pred = model(x_batch).view(-1)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            #scheduler.step()
            optimizer.zero_grad()
            with torch.no_grad():
                loss_at_train += (torch.square((y_pred - y_batch)).sum()).item()
        rmse_at_train = (loss_at_train / len(train_loader.dataset)) ** (1/2) * scalerY.scale_[0]
        metrics_dict["Train RMSE"].append(rmse_at_train)

        loss_at_valid = 0
        for batch_v in valid_loader:
            x_batch_v, y_batch_v = batch_v
            x_batch_v, y_batch_v = x_batch_v.to(DEVICE), y_batch_v.to(DEVICE)
            y_pred_v = model(x_batch_v).view(-1)
            with torch.no_grad():
                loss_at_valid += (torch.square((y_pred_v - y_batch_v)).sum()).item()
        rmse_at_valid = (loss_at_valid / len(valid_loader.dataset)) ** (1/2) * scalerY.scale_[0]
        metrics_dict["Valid RMSE"].append(rmse_at_valid)
        
        loss_at_test = 0
        for batch_t in test_loader:
            x_batch_t, y_batch_t = batch_t
            x_batch_t, y_batch_t = x_batch_t.to(DEVICE), y_batch_t.to(DEVICE)
            y_pred_t = model(x_batch_t).view(-1)
            with torch.no_grad():
                loss_at_test += (torch.square((y_pred_t - y_batch_t)).sum()).item()
        rmse_at_test = (loss_at_test / len(test_loader.dataset)) ** (1/2) * scalerY.scale_[0]
        metrics_dict["Test RMSE"].append(rmse_at_test)

        clear_output(wait=True)
        display(pd.DataFrame(metrics_dict))
        
    plt.plot(metrics_dict['Epoch'], metrics_dict['Train RMSE'], 'darkorchid', label='Train', linewidth=2)
    plt.plot(metrics_dict['Epoch'], metrics_dict['Valid RMSE'], 'crimson', label='Valid', linewidth=2)
    plt.plot(metrics_dict['Epoch'], metrics_dict['Test RMSE'], 'green', label='test', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend(shadow=False, fontsize=14)
    plt.show()
    
def test(model, criterion, test_loader, test_set, DEVICE='cpu'):
    '''
    returns:
        predicts - torch.tensor with shape (len(test_loader.dataset), ),
                   which contains predictions for test objects
    '''
    predicts = torch.ones(len(test_loader.dataset))
    loss_at_test = 0
    for batch in test_loader:
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        y_pred = model(x_batch).view(-1)
        loss = criterion(y_pred, y_batch)
        with torch.no_grad():
            loss_at_test += (torch.square((y_pred - y_batch)).sum()).item()
    rmse_at_test = (loss_at_test / len(test_loader.dataset)) ** (1/2) * scalerY.scale_[0]
    print(rmse_at_test)
    
    return model(test_set.data)
