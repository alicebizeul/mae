import os
import logging
from datetime import datetime
from transformers import ViTMAEConfig, ViTMAEModel, ViTMAEForPreTraining
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
from PIL import Image
from dataset import PCA, IndexDatasetWrapper, LabelSubsetDataset, RepDataset
import argparse
import numpy as np
import torchvision
from networks import get_configs, ResNetAutoEncoder
from utils import save_reconstructed_images, save_attention_maps, get_eigenvalues, LinearWarmupScheduler, PairedImageDataset
import random
import matplotlib.pyplot as plt 
import time 
import csv
from plotting import plot_loss, plot_performance
import torch.nn.functional as F

# from lars import LARS


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Script to train a model with custom parameters.')
    parser.add_argument('--dataset', type=str, default="tinyimagenet", help='Number of epochs to train the model.')    
    parser.add_argument('--dataset2', type=str, default="tinyimagenet", help='Number of epochs to train the model.') 
    parser.add_argument('--dataset_og', type=str, default="tiny-imagenet-200", help='Number of epochs to train the model.')       
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model.')
    parser.add_argument('--eval_epochs', type=int, default=200, help='Number of epochs to train the model.')
    parser.add_argument('--eval_freq', type=int, default=500, help='Number of epochs to train the model.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--batch_size_eval', type=int, default=4096, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=1.5e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--mask-ratio', type=float, default=0.75, help='Learning rate for the optimizer.')
    parser.add_argument('--ratio_pc', type=float, default=0.94, help='Ratio of principal components to retain.')
    parser.add_argument('--type', type=str, default="ae_mask",choices=["ae","pca_ae","ae_mask"],help='Ratio of principal components to retain.')
    parser.add_argument('--arch', type=str, default="vit_t",choices=['vit_t','vit_s'],help='Ratio of principal components to retain.')    
    parser.add_argument('--save_dir', type=str, default="/local/home/abizeul/reconstruction/outputs",help='Ratio of principal components to retain.')    
    parser.add_argument('--run_id', type=str, default="ae",help='Ratio of principal components to retain.')    
    parser.add_argument('--seed', type=int, default=0, help='Batch size for training.')
    parser.add_argument('--debug', action="store_true", help='Debug round with subsample.')
    parser.add_argument('--small_scale', action="store_true", help='Debug round with subsample.')
    parser.add_argument('--eval', action="store_true", help='Debug round with subsample.')
    parser.add_argument('--pretrained', default=None, help="Weights")
    parser.add_argument('--root', type=str, default="/local/cluster/abizeul/data")

    args = parser.parse_args()

    if "tinyimagenet" in args.dataset:
        args.n_channels=3
        args.resolution=64
        args.data_dim = args.n_channels*(args.resolution**2)
    
    # Calculate derived parameter
    # args.nb_pc = int(args.ratio_pc * args.data_dim)
    return args

# # Define the training function
# def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
#     model.to(device)
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         for images, _ in train_loader:
#             images = images.to(device)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, images)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()

#         epoch_loss = running_loss / len(train_loader)
#         logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Main function
def main():
    # Hyperparameters
    args = parse_args()

    # setting seeds 
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # creating folders
    if args.run_id is None:
        args.save_dir = os.path.join(args.save_dir, args.run_id, "ViT", str(args.seed), str(args.mask_ratio))
    else:
        args.save_dir = os.path.join(args.save_dir, args.run_id, "ViT", str(args.seed), str(args.mask_ratio))
    os.makedirs(args.save_dir,exist_ok=True)
    os.makedirs(os.path.join(args.save_dir,"networks"),exist_ok=True)
    os.makedirs(os.path.join(args.save_dir,"eigenvalues"),exist_ok=True)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data transforms
    transform = transforms.ToTensor()

    # Data
    trainset = PairedImageDataset(folder_A=f'{args.root}/{args.dataset}/train', folder_B=f'{args.root}/{args.dataset2}/train', transform=transform)
    trainset_eval = torchvision.datasets.ImageFolder(f'{args.root}/{args.dataset}/train', transform=transform)
    trainset_eval2 = torchvision.datasets.ImageFolder(f'{args.root}/{args.dataset_og}/train', transform=transform)

    valset   = torchvision.datasets.ImageFolder(f'{args.root}/{args.dataset}/val', transform=transform)
    valset2 = torchvision.datasets.ImageFolder(f'{args.root}/{args.dataset_og}/val', transform=transform)

    if args.debug:
        subset_indices = torch.randperm(len(trainset))[:10*args.batch_size]
        trainset = Subset(trainset, subset_indices)
    elif args.small_scale:
        subset_indices = torch.randperm(len(trainset))[:int(0.5*len(trainset))]
        trainset = Subset(trainset, subset_indices)

    trainloader =  torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16, drop_last=True)
    trainloader_single=  torch.utils.data.DataLoader(trainset_eval, batch_size=args.batch_size, shuffle=True, num_workers=16, drop_last=True)
    trainloader_og=  torch.utils.data.DataLoader(trainset_eval2, batch_size=args.batch_size, shuffle=True, num_workers=16, drop_last=True)
    valloader  =  torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=16, drop_last=True)
    valloader_og  =  torch.utils.data.DataLoader(valset2, batch_size=args.batch_size, shuffle=False, num_workers=16, drop_last=True)

    # Model, criterion, and optimizer
    # config, bottleneck = get_configs(arch=args.arch)
    # if args.type != "ae_mask":
    #     config = ViTMAEConfig(hidden_size=192,num_attention_head=3,intermediate_size=768,image_size=64,patch_size=8,mask_ratio=0.0,norm_pix_loss=True)
    # else:

    if args.arch == "vit_t":
        config = ViTMAEConfig(hidden_size=192,num_attention_head=3,intermediate_size=768,image_size=64,patch_size=8,mask_ratio=args.mask_ratio,norm_pix_loss=True,attn_implementation="eager")
    elif args.arch == "vit_s":
        config = ViTMAEConfig(hidden_size=384,num_attention_head=6,intermediate_size=1536,image_size=64,patch_size=8,mask_ratio=args.mask_ratio,norm_pix_loss=True,attn_implementation="eager")
    else: raise NotImplementedError

    model = ViTMAEForPreTraining(config).to(device)
    if args.pretrained is not None:
        try:
            print(model.load_state_dict(torch.load(args.pretrained).state_dict()),flush=True)
            model.load_state_dict(torch.load(args.pretrained).state_dict())
        except:
            print(model.load_state_dict(torch.load(args.pretrained)).state_dict())
            print("Loading of weights did not work",flush=True)

    latent_dim = 192 if args.arch == "vit_t" else 384 if args.arch== "vit_s" else 192 # HARDCODED
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.05)
    scheduler = LinearWarmupScheduler(optimizer, 10, args.epochs, args.learning_rate)

    # eval classifier ALICE: YOU NEED TO ADJUST THIS TO PAPER
    head = nn.Linear(latent_dim,200).to(device)
    head_og = nn.Linear(latent_dim,200).to(device)
    optimizer_head = optim.AdamW(head.parameters(), lr=1e-2)
    optimizer_head_og = optim.AdamW(head_og.parameters(), lr=1e-2)
    # optimizer_head = optim.SGD(head.parameters(), lr=0.1, weight_decay=0.0, momentum=0.9) # larger batch size: 4096
    scheduler_head = LinearWarmupScheduler(optimizer_head, 10, args.eval_epochs, 1e-2)
    scheduler_head_og = LinearWarmupScheduler(optimizer_head_og, 10, args.eval_epochs, 1e-2)
    loss_fn = nn.CrossEntropyLoss()

    # logging
    loss_values, loss_tracking, loss_values_eval, loss_values_eval_og, loss_tracking_eval, loss_tracking_eval_og = [], [], [], [], [], []
    scores, scores_og = {}, {}
    time_batch=[]
    for epoch in range(args.epochs):
        if not args.eval:
            model.train().to("cuda")
            t0 = time.time()
            t0_1 = time.time()
            for batch_index, (batch1, batch2) in enumerate(trainloader):
                
                optimizer.zero_grad()

                batch1 = batch1.to(device)
                batch2 = batch2.to(device)

                outputs = model(batch1)
                reconstruction = model.unpatchify(outputs.logits)
                
                if mask_ratio > 0.0:
                    mask = outputs.mask.unsqueeze(-1).repeat(1, 1, model.config.patch_size**2 *3)  # (N, H*W, p*p*3)
                    mask = model.unpatchify(mask)
                    if epoch==0 and batch_index==0: print("Sum of mask",torch.sum(mask))
                    loss = criterion(mask*batch2,mask*reconstruction)
                else:
                    loss = criterion(batch2,reconstruction)
                # else:
                #     reconstruction = model(batch)
                #     loss = criterion(batch,reconstruction)
                loss.backward()
                optimizer.step()

                loss_values.append(np.sqrt(loss.item()))
                average_loss = np.mean(loss_values)
                loss_tracking.append(average_loss)
                
                if (epoch%args.eval_freq==0 or (epoch+1)==args.epochs) and batch_index==0:
                    if mask_ratio > 0:
                        save_reconstructed_images((-1*(mask[:10]-1))*batch1[:10],(-1*(mask[:10]-1))*batch2[:10], reconstruction[:10], epoch, args.save_dir,"train")
                    else:
                        save_reconstructed_images(batch1[:10], batch2[:10], reconstruction[:10], epoch, args.save_dir,"train")

                # time_batch.append(time.time()-t0_1)
                # t0_1=time.time()
                # print(batch_index,"/",len(trainloader),np.mean(time_batch))

            # plotting
            plot_loss(loss_tracking,name_loss="RMSE",dir=args.save_dir,name_file="_train")
            if (epoch+1)%10==0:
                print(f"---- Epoch {epoch}/{args.epochs} in {round(time.time()-t0,2)}s with {round(np.mean(loss_tracking),2)} loss",flush=True)
            scheduler.step(epoch)

        if epoch%args.eval_freq==0 or (epoch+1)==args.epochs:
            print("Started eval",flush=True)
            model.eval().to("cpu")

            ## compute dict for model class change
            new_state_dict = {}
            for key, value in model.state_dict().items():
                new_key = key.replace("vit.", "")
                new_state_dict[new_key] = value

            # define evaluation model
            model_eval = ViTMAEModel(config)
            model_eval.load_state_dict(new_state_dict,strict=False)
            model_eval=model_eval.to(device)
            del new_state_dict

            # collect representations 
            representations_list, labels_list, attentions_list, images_list = [], [], [], []
            with torch.no_grad():
                for _, (batch, labels) in enumerate(trainloader_single):
                    batch = batch.to(device)
                    outputs = model_eval(batch,output_attentions=True) #[0][:,0].reshape(batch.shape[0],-1).detach().cpu()
                    attentions = outputs.attentions[-1].mean(1).detach().cpu()
                    outputs = outputs.last_hidden_state[:,0].reshape(batch.shape[0],-1).detach().cpu()

                    for i, l, r, a in zip(batch, labels,outputs,attentions):
                        representations_list.append(r)
                        attentions_list.append(a)
                        labels_list.append(l)
                        images_list.append(i.detach().cpu())

            trainset = RepDataset(images_list, representations_list,labels_list, attentions_list)
            trainloader_eval = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_eval, shuffle=False, num_workers=8, drop_last=True)

            representations_list, labels_list, attentions_list, images_list = [], [], [], []
            with torch.no_grad():
                for _, (batch, labels) in enumerate(valloader):
                    batch = batch.to(device)
                    outputs = model_eval(batch,output_attentions=True) #[0][:,0].reshape(batch.shape[0],-1).detach().cpu()
                    attentions = outputs.attentions[-1].mean(1).detach().cpu()
                    outputs = outputs.last_hidden_state[:,0].reshape(batch.shape[0],-1).detach().cpu()

                    for i, l, r, a in zip(batch, labels,outputs,attentions):
                        representations_list.append(r)
                        attentions_list.append(a)
                        labels_list.append(l)
                        images_list.append(i.detach().cpu())

            valset = RepDataset(images_list, representations_list,labels_list, attentions_list)
            valloader_eval  =  torch.utils.data.DataLoader(valset, batch_size=args.batch_size_eval, shuffle=False, num_workers=8, drop_last=True)

            # if we are at the last epoch than we want to reinitialize the head and optimizer
            if (epoch+1)==args.epochs:
                head = nn.Linear(latent_dim,200).to(device)
                optimizer_head = optim.AdamW(head.parameters(), lr=1e-2)
                scheduler_head = LinearWarmupScheduler(optimizer_head, 10, args.eval_epochs, 1e-2)

            # final evaluation
            for eval_epoch in range(args.eval_epochs):
                if (eval_epoch+1)%10==0:print(f"---- Eval Epoch {eval_epoch+1}/{args.eval_epochs}")
                for batch_index, (imgs, batch, labels, atts) in enumerate(trainloader_eval):

                    #prep
                    optimizer_head.zero_grad()
                    batch = batch.to(device)
                    labels = labels.to(device)

                    # forward
                    logits = head(batch)

                    # loss
                    loss = loss_fn(logits,labels)

                    # backward
                    loss.backward()
                    optimizer_head.step()

                    loss_values_eval.append(loss.item())

                    if eval_epoch == 0 and batch_index==0:
                        cls_token_map = atts[:,0,1:].reshape([-1,8,8])
                        att_map = F.interpolate(cls_token_map.unsqueeze(1),size=(64,64),mode='bilinear', align_corners=False)
                        save_attention_maps(imgs[:10],att_map[:10],epoch, args.save_dir,"eval")

                loss_tracking_eval.append(np.mean(loss_values_eval))
                scheduler_head.step(eval_epoch)
                
                # plot loss
                plot_loss(loss_tracking_eval,name_loss="XEnt",dir=args.save_dir,name_file="_eval")
                    
            score = 0
            with torch.no_grad():
                for _, (imgs, batch, labels, atts) in enumerate(valloader_eval):
                    batch = batch.to(device)
                    prediction = torch.argmax(head(batch),dim=-1).detach().cpu()
                    score += sum(1*(prediction.numpy() == labels.detach().cpu().numpy()))
                score /= len(valloader_eval)*args.batch_size_eval
                score *= 100
            scores[epoch]=round(score,2)
            # plot
            plot_performance(list(scores.keys()),list(scores.values()),args.save_dir)

            if (epoch+1)==args.epochs:

                # collect representations 
                representations_list, labels_list, attentions_list, images_list = [], [], [], []
                with torch.no_grad():
                    for _, (batch, labels) in enumerate(trainloader_og):
                        batch = batch.to(device)
                        outputs = model_eval(batch,output_attentions=True,return_dict=True) #[0][:,0].reshape(batch.shape[0],-1).detach().cpu()
                        attentions = outputs.attentions[-1].mean(1).detach().cpu()
                        outputs = outputs.last_hidden_state[:,0].reshape(batch.shape[0],-1).detach().cpu()

                        for i, l, r, a in zip(batch, labels,outputs,attentions):
                            representations_list.append(r)
                            attentions_list.append(a)
                            labels_list.append(l)
                            images_list.append(i.detach().cpu())

                trainset = RepDataset(images_list, representations_list,labels_list, attentions_list)
                trainloader_eval = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_eval, shuffle=False, num_workers=8, drop_last=True)

                representations_list, labels_list, attentions_list, images_list = [], [], [], []
                with torch.no_grad():
                    for _, (batch, labels) in enumerate(valloader_og):
                        batch = batch.to(device)
                        outputs = model_eval(batch,output_attentions=True)
                        attentions=outputs.attentions[-1].mean(1).detach().cpu()
                        outputs = outputs.last_hidden_state[:,0].reshape(batch.shape[0],-1).detach().cpu()

                        for i, l, r, a in zip(batch, labels,outputs,attentions):
                            representations_list.append(r)
                            labels_list.append(l)
                            attentions_list.append(a)
                            images_list.append(i.detach().cpu())

                valset = RepDataset(images_list, representations_list, labels_list, attentions_list)
                valloader_eval  =  torch.utils.data.DataLoader(valset, batch_size=args.batch_size_eval, shuffle=False, num_workers=8, drop_last=True)

                # final evaluation
                for eval_epoch in range(args.eval_epochs):
                    if (eval_epoch+1)%10==0:print(f"---- OG Eval Epoch {eval_epoch+1}/{args.eval_epochs}")
                    for batch_index, (imgs, batch, labels, atts) in enumerate(trainloader_eval):

                        #prep
                        optimizer_head_og.zero_grad()
                        batch = batch.to(device)
                        labels = labels.to(device)

                        # forward
                        logits = head_og(batch)

                        # loss
                        loss = loss_fn(logits,labels)

                        # backward
                        loss.backward()
                        optimizer_head_og.step()

                        loss_values_eval_og.append(loss.item())

                        if eval_epoch == 0 and batch_index==0:
                            cls_token_map = atts[:,0,1:].reshape([-1,8,8])
                            att_map = F.interpolate(cls_token_map.unsqueeze(1),size=(64,64),mode='bilinear', align_corners=False)
                            save_attention_maps(imgs[:10],att_map[:10],epoch, args.save_dir,"eval_og")


                    loss_tracking_eval_og.append(np.mean(loss_values_eval_og))
                    scheduler_head_og.step(eval_epoch)
                    
                    # plot loss
                    plot_loss(loss_tracking_eval_og,name_loss="XEnt",dir=args.save_dir,name_file="_eval_og")
                    
                score = 0
                with torch.no_grad():
                    for _, (imgs, batch, labels, atts) in enumerate(valloader_eval):
                        batch = batch.to(device)
                        prediction = torch.argmax(head_og(batch),dim=-1).detach().cpu()
                        score += sum(1*(prediction.numpy() == labels.detach().cpu().numpy()))
                    score /= len(valloader_eval)*args.batch_size_eval
                    score *= 100
                scores_og[epoch]=round(score,2)
                # plot
                plot_performance(list(scores_og.keys()),list(scores_og.values()),args.save_dir,name="og")

            # going back
            del model_eval
            model.train().to(device)

            # save
            if not args.eval: torch.save(model,os.path.join(args.save_dir,"networks",f"model_{epoch}_epochs.pth")) 

    # save
    torch.save(model,os.path.join(args.save_dir,"networks",f"model_{args.epochs}_epochs.pth")) 

    # Write to a CSV file
    with open(os.path.join(args.save_dir,'performance_final.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Eval Epoch', 'Test Accuracy'])
        for epoch in list(scores.keys()):
            # Assuming you have the accuracy for each epoch stored in a list
            writer.writerow([epoch+1, round(scores[epoch],2)])  

    # Write to a CSV file
    with open(os.path.join(args.save_dir,'performance_final_og.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Eval Epoch', 'Test Accuracy'])
        for epoch in list(scores_og.keys()):
            # Assuming you have the accuracy for each epoch stored in a list
            writer.writerow([epoch+1, round(scores_og[epoch],2)]) 

if __name__ == '__main__':
    main()