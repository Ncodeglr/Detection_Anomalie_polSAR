
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAELoss(nn.Module):
    def __init__(self, beta_scheduler):
        """
        Initialise la fonction de perte VAE avec un planificateur de beta.

        Paramètres :
        ------------
        beta_scheduler : callable
            Fonction qui donne la valeur de beta en fonction de l'epoch.
        """
        super(VAELoss, self).__init__()
        self.beta_scheduler = beta_scheduler

    def forward(self, x, recon_x, mu, log_var, epoch):
        """
        Calcule la perte VAE.

        Paramètres :
        ------------
        x : torch.Tensor
            Données d'entrée originales.
        recon_x : torch.Tensor
            Données reconstruites.
        mu : torch.Tensor
            Moyenne de l'espace latent.
        log_var : torch.Tensor
            Log-variance de l'espace latent.
        epoch : int
            Epoch actuelle pour ajuster beta.

        Retourne :
        ----------
        torch.Tensor : La perte totale.
        """
        # Perte de reconstruction
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        # Perte KL
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        beta = self.beta_scheduler(epoch)
        rate_loss = recon_loss / kl_loss
        return recon_loss + beta * kl_loss, rate_loss, recon_loss, kl_loss

class VAELoss_complex(nn.Module):
    """
    Initializes the ComplexVAE Loss module.

    Parameters:
    ----------
    beta : float
        Weighting factor for the KL divergence term. Default is 1.0.
    """
    def __init__(self, beta=1.0):
        super(VAELoss_complex, self).__init__()
        self.beta = beta

    def forward(self, original, reconstructed, mu, sigma, delta):
        """
        Computes the loss for the ComplexVAE.

        Parameters:
        ----------
        original : torch.Tensor
            Original complex input tensor.
        reconstructed : torch.Tensor
            Reconstructed complex tensor from the decoder.
        mu : torch.Tensor
            Mean vector of the latent space distribution from the encoder.
        sigma : torch.Tensor
            Variance vector of the latent space distribution from the encoder.
        delta : torch.Tensor
            Pseudo-variance vector of the latent space distribution from the encoder.

        Returns:
        -------
        tuple
            Total loss (real scalar), reconstruction loss, KL divergence loss.
        """
        # Separate real and imaginary parts
        original_real, original_imag = original.real, original.imag
        reconstructed_real, reconstructed_imag = reconstructed.real, reconstructed.imag

        # Reconstruction loss (real L2 loss for each component)
        recon_loss_real = torch.sum((reconstructed_real - original_real) ** 2)
        recon_loss_imag = torch.sum((reconstructed_imag - original_imag) ** 2)
        reconstruction_loss = recon_loss_real + recon_loss_imag

        # KL divergence loss (use real components)
        kl_divergence = (
            torch.sum(mu.real**2 + mu.imag**2) +
            torch.sum(sigma.real - 1 - 0.5 * torch.log(sigma.real**2 - torch.abs(delta)**2))
        )

        # Total loss
        total_loss = reconstruction_loss + self.beta * kl_divergence

        return total_loss, reconstruction_loss, kl_divergence, mu, sigma, delta

def train_epoch(model, dataloader, optimizer, beta, device, epoch):
    """
    Train the model for one epoch.
    
    Parameters:
    ----------
    model : torch.nn.Module
        The VAE model to be trained.
    dataloader : torch.utils.data.DataLoader
        DataLoader providing the training data.
    optimizer : torch.optim.Optimizer
        Optimizer used for training.
    loss_function : callable
        The loss function (VAELoss in this case).
    device : str
        Device on which the model is trained ('cuda' or 'cpu').
    
    Returns:
    -------
    tuple
        Averages of the total loss, reconstruction loss, KL divergence loss, and rate loss over the entire epoch.
    """
    loss_function = VAELoss_complex(beta)
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_rate_loss = 0

    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass through the model
        reconstructed_data, mu, logvar = model(data)
        
        # Compute the loss
        loss, rate_loss, recon_loss, kl_loss = loss_function(data, reconstructed_data, mu, logvar, epoch)
        loss.backward()  # Backpropagation
        
        # Update the model parameters
        optimizer.step()
        
        # Accumulate the losses
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        total_rate_loss += rate_loss.item()

        # Clip gradients to avoid explosion of gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Calculate the average losses for the epoch
    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_kl_loss = total_kl_loss / len(dataloader)
    avg_rate_loss = total_rate_loss / len(dataloader)

    return avg_loss, avg_rate_loss, avg_recon_loss, avg_kl_loss


def test_epoch(model, dataloader, loss_function, device, epoch):
    """
    Evaluate the model on the test set.
    
    Parameters:
    ----------
    model : torch.nn.Module
        The VAE model to be evaluated.
    dataloader : torch.utils.data.DataLoader
        DataLoader providing the test data.
    loss_function : callable
        The loss function (VAELoss in this case).
    device : str
        Device on which the model is evaluated ('cuda' or 'cpu').
    
    Returns:
    -------
    tuple
        Averages of the total loss, reconstruction loss, KL divergence loss, and rate loss over the entire test set.
    """
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_rate_loss = 0

    with torch.no_grad():  # Disable gradient computation for faster evaluation
        for data, _ in dataloader:
            data = data.to(device)
            
            # Forward pass
            reconstructed_data, mu, logvar = model(data)
            
            # Compute the loss
            loss, rate_loss, recon_loss, kl_loss = loss_function(data, reconstructed_data, mu, logvar, epoch)

            # Accumulate the losses
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_rate_loss += rate_loss.item()

    # Calculate average losses for the test set
    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_kl_loss = total_kl_loss / len(dataloader)
    avg_rate_loss = total_rate_loss / len(dataloader)

    return avg_loss, avg_rate_loss, avg_recon_loss, avg_kl_loss

def train(model, loader, cfg):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"]
    )

    device = cfg["misc"]["device"]
    model.train()

    for epoch in range(cfg["train"]["epochs"]):

        epoch_loss = 0

        for x,_ in loader:

            x = x.to(device)

            optimizer.zero_grad()

            x_hat = model(x)

            #loss = torch.mean((x - x_hat)**2)
            loss = torch.mean(torch.abs(x_hat - x) ** 2)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print("epoch", epoch, "loss", epoch_loss/len(loader))

# def train(model, loader, cfg):
#     train_losses = []
#     train_recon_losses = []
#     train_kl_losses = []
#     train_rate_losses = []

#     test_losses = []
#     test_recon_losses = []
#     test_kl_losses = []
#     test_rate_losses = []
#     optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]['learning_rate'])

#     for epoch in range(cfg["train"]["epochs"]):
#         train_loss, train_rate, train_recon_loss, train_kl_loss = train_epoch(
#             model, loader, model.optimizer, model.objective, cfg["misc"]["device"], epoch)

#         train_losses.append(train_loss)
#         train_recon_losses.append(train_recon_loss)
#         train_kl_losses.append(train_kl_loss)
#         train_rate_losses.append(train_rate)

#     #return train_losses, test_losses, test_rate_losses, test_recon_losses, test_kl_losses
#     return train_losses