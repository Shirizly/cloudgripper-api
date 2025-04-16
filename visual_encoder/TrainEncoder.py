import torch



def encoder_training(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, weight_path="model_weights.pth"):
    """
    Train the encoder model using the given data loaders and hyperparameters.
    :param model: Encoder model
    :param train_loader: DataLoader for training data
    :param val_loader: DataLoader for validation data
    :param criterion: Loss function
    :param optimizer: Optimizer
    :param num_epochs: Number of epochs to train the model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # saving loss data
    train_losses = []
    val_losses = []
    model.train()  # Set model to training mode
    scaler = torch.amp.GradScaler('cuda')
    # Iterate through batches
    for epoch in range(num_epochs):
        train_loss = 0.0
        
        for images, metadata in train_loader:
            
            images, metadata = images.to(device), metadata.to(device)  # Move batch to GPU
            
            # Display one image from the batch using cv2
            # img = images[0].permute(1, 2, 0).cpu().numpy()  # Convert tensor to numpy array and change dimensions
            # img = (img * 255).astype('uint8')  # Denormalize and convert to uint8
            # cv2.imshow('Image', img)
            # cv2.waitKey(2000)  # Display the image for 1 ms
            
            optimizer.zero_grad()  # Reset gradients
            
            # Forward pass
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                # Compute loss (mean squared error over the first 5 values of metadata)
                # loss = criterion(outputs, metadata)
                loss = criterion(outputs[:,:3], metadata[:,:3]) # only use the first 3 values of metadata
            # Backpropagation
            # loss.backward()
            # optimizer.step()

            # backpropagate with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))
        model.eval()  # Set model to evaluation mode    
        val_loss = 0.0
        with torch.no_grad():
            for images, metadata in val_loader:
                images, metadata = images.to(device), metadata.to(device)  # Move batch to GPU
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    # Compute loss (mean squared error over the first 5 values of metadata)
                    # loss = criterion(outputs, metadata)
                    loss = criterion(outputs[:,:3], metadata[:,:3]) # only use the first 3 values of metadata
            
                val_loss += loss.item()
        
        val_losses.append(val_loss / len(val_loader))
        
        print(f"Epoch {epoch+1}/{num_epochs}, Training loss: {train_losses[-1]:.4f}, Validation loss: {val_losses[-1]:.4f}")

        # if epoch % 10 == 0:
        #     # Save model checkpoint
        #     torch.save(model.state_dict(), f"model_checkpoint_{epoch}.pth")

    print("Training complete.")

    # Save the trained model
    torch.save(model.state_dict(), weight_path)

    return train_losses, val_losses