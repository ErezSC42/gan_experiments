import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#  my models
from models.Generator import Generator
from models.Discriminator import Discriminator


if __name__ == '__main__':
    #hyperparameters
    lr = 0.0002
    batch_size = 64
    image_size = 64  #mnist has 28,28 we will need to resize to 64,64
    channels_num = 1
    noise_dim = 256
    epochs = 10
    hidden_dim_dis = 16
    hidden_dim_gen = 16

    preprocessing_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,)),
    ])

    dataset = datasets.MNIST(root="data/", train=True, transform=preprocessing_transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    discriminator_model = Discriminator(channels_num, hidden_dim_dis)
    generator_model = Generator(noise_dim, channels_num, hidden_dim_gen)

    optimizer_dis = optim.Adam(discriminator_model.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_gen = optim.Adam(generator_model.parameters(), lr=lr, betas=(0.5, 0.999))

    discriminator_model.train().cuda()
    generator_model.train().cuda()

    criterion = nn.BCELoss()

    label_fake = 0
    label_real = 1
    fixed_noise = torch.randn(64, noise_dim, 1, 1).to(device)

    writer_real = SummaryWriter(f"runs/gan_mnist/test_real")
    writer_fake = SummaryWriter(f"runs/gan_mnist/test_fake")

    for epoch in tqdm.trange(epochs):
        for batch_idx, (data, labels) in enumerate(dataloader):
            data = data.to(device)
            batch_size = data.shape[0]
            discriminator_model.zero_grad()
            generator_model.zero_grad()
            ### train discrimiator: max(log(D(x)) + log(1 - D(G(Z))

            # first, we train on real images, so we set labels to be 1
            # trick - set labels to 0.9, to make the discriminator be not entirely sure that these are the real images
            label = (torch.ones(batch_size) * 0.9).to(device)
            output = discriminator_model(data).reshape(-1)  # vector of probabilities
            loss_discriminator_real = criterion(output, label)
            discriminator_x = output.mean().item()  # average confidence, for tracking

            #  train discriminator on fake images
            noise = torch.randn(batch_size, noise_dim, 1, 1).to(device)
            fake_images_batch = generator_model(noise)
            # trick - set labels to 0.1, to make the discriminator be not entirely sure that these are the fake images
            label = (torch.ones((batch_size)) * 0.1).to(device)
            output = discriminator_model(fake_images_batch.detach()).reshape(-1) # detach G gradients, training D only
            loss_discriminator_fake = criterion(output, label)

            loss_discriminator = loss_discriminator_fake + loss_discriminator_real
            loss_discriminator.backward()
            optimizer_dis.step()

            ### Train Generator - max log(D(G(z)))
            label = torch.ones(batch_size).to(device)
            output = discriminator_model(fake_images_batch).reshape(-1)
            loss_generator = criterion(output, label)
            loss_generator.backward()
            optimizer_gen.step()

            if batch_idx % 100 == 0:
                print(f"epoch [{epoch}/{epochs}] Batch [{batch_idx}/{len(dataloader)}] \
                Loss D: {loss_discriminator:.4f}, lossG {loss_generator:.4f}, D(x):{discriminator_x:.4f}")

            with torch.no_grad():
                fake = generator_model(fixed_noise)
                img_grid_real = make_grid(data[:32], normalize=True)
                img_grid_fake = make_grid(fake[:32], normalize=True)
                writer_real.add_image("MNIST real Image: ", img_grid_real)
                writer_fake.add_image("MNIST fake Image: ", img_grid_fake)




