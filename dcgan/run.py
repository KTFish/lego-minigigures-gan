import os
import torch
from only_lego_dataset import get_dataloader
from dataset import get_dataloaders
import config as c
from generator import Generator
from discriminator import Discriminator
import utils
from utils import weights_init, sample_noise, save_tensor_images
from tqdm import tqdm

utils.setup_generated_image_folders()

# ! dataloader, _ = get_dataloaders(c.BATCH_SIZE)
dataloader = get_dataloader()

# ! gen = Generator(c.Z_DIM).to(c.DEVICE)
from generator300 import Generator300
from discriminator300 import Discriminator300
gen = Generator300().to(c.DEVICE)
gen_opt = torch.optim.Adam(gen.parameters(), lr=c.LEARNING_RATE, betas=c.BETAS)
# !disc = Discriminator().to(c.DEVICE)
disc = Discriminator300().to(c.DEVICE)
disc_opt = torch.optim.Adam(disc.parameters(), lr=c.LEARNING_RATE, betas=c.BETAS)
gen = gen.apply(weights_init)
disc = disc.apply(weights_init)

curr_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
criterion = c.CRITERION
display_step=c.DISPLAY_STEP

for epoch in range(c.NUM_EPOCHS):
    # Dataloader returns the batches
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)
        real = real.to(c.DEVICE)

        ## Update discriminator ##
        disc_opt.zero_grad()
        fake_noise = sample_noise(cur_batch_size, c.Z_DIM, device=c.DEVICE)
        fake = gen(fake_noise)
        disc_fake_pred = disc(fake.detach())
        disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_pred = disc(real)
        disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2

        # Keep track of the average discriminator loss
        mean_discriminator_loss += disc_loss.item() / display_step
        # Update gradients
        disc_loss.backward(retain_graph=True)
        # Update optimizer
        disc_opt.step()

        ## Update generator ##
        gen_opt.zero_grad()
        fake_noise_2 = sample_noise(cur_batch_size, c.Z_DIM, device=c.DEVICE)
        fake_2 = gen(fake_noise_2)
        disc_fake_pred = disc(fake_2)
        gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        gen_loss.backward()
        gen_opt.step()

        # Keep track of the average generator loss
        mean_generator_loss += gen_loss.item() / display_step

        
        if curr_step % display_step == 0 and curr_step > 0:
            utils.print_training_progress(epoch, curr_step, mean_generator_loss, mean_discriminator_loss)
            # print(
            #     f"Epoch {epoch}, step {curr_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}"
            # )
            save_tensor_images(fake, img_type='fake', epoch=epoch, step=curr_step)
            save_tensor_images(real, img_type='real', epoch=epoch, step=curr_step)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
            utils.save_model_checkpoint(gen, name=f"gen300-{epoch}", path="./models/dcgan300/")
            utils.save_model_checkpoint(disc, name=f"disc300-{epoch}", path="./models/dcgan300/")
        curr_step += 1