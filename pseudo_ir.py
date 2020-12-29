def algorithm2_random(data_all, dname, network_name, batch_size, num_epochs, filtered=True): # Uses same model
    train, val, trainloader,valloader,testloader = data_all # get_dataset(batch_size, dname, filtered) # get_mnist(batch_size)
    subsample_size = round(len(train)/5) # 5 iterations in total 7500/5 = 1500
    ensemble = {}
    if dname == "cifar10" and filtered==True:
        num_classes = 2
    else:
        num_classes = 10
    # set seed
    train_sub, _ = torch.utils.data.random_split(train,[subsample_size,len(train)-subsample_size])
    tr_sub_ld = torch.utils.data.DataLoader(train_sub, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=1)
    model, val_loss, train_loss = train_and_eval_model(network_name, dname, tr_sub_ld, valloader, batch_size, num_epochs) # don't use full dataset
    ensemble[model] = val_loss
    ensemble_vals = [val_loss]
    val_losses = [val_loss]
    train_losses = [train_loss]
    test_acc = get_ensemble_preds(ensemble, testloader,"test")
    test_accs = [test_acc]
    data_inds = []
    for i in range(4):
        poor_subset_loader, indices = get_poor_subset(ensemble, trainloader, train, batch_size, subsample_size, num_classes)
        model, val_loss, train_loss = train_and_eval_model(network_name, dname, poor_subset_loader, valloader, batch_size, num_epochs, trained_model = model)
        ens_acc = get_ensemble_preds(ensemble, valloader,"validation")
        data_inds.extend(indices)
        val_losses.append(val_loss)   # do we want to pick or weigh based on val_loss?
        train_losses.append(train_loss)
        ensemble[model] = val_loss
        ensemble_vals.append(ens_acc)
        test_acc = get_ensemble_preds(ensemble, testloader,"test")
        test_accs.append(test_acc)
    data_prop = (len(data_inds)+subsample_size)/len(train)*100
    print(test_acc)
    print("data percentage used",data_prop)
    return train_losses, val_losses, ensemble_vals, test_accs, data_prop
