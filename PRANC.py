import os
import math
import time
import torch
import random
import argparse
import torchvision
import util
import tqdm
import imageio
from collections import OrderedDict
from util import get_clamped_psnr
from torch import nn
from torch import optim
from siren import Siren
from torchvision import transforms
from torchvision.utils import save_image
from training import Trainer


def save_signature(dirname, mean = None, var = None):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    torch.save(alpha, dirname + '/lr.pt')
    if 'resnet' in args.model:
        torch.save(mean, dirname + '/means.pt')
        torch.save(var, dirname + '/vars.pt')
        



parser = argparse.ArgumentParser(description='Arguments of program')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--size', default=32, type=int)
parser.add_argument('--k', default=10000, type=int)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--epoch', default=1, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--window', default=500, type=int, help="The number of alphas in each coordinate descent (the m in paper)")
parser.add_argument('--log-rate', default=50, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--dataset', default='../datasets', type=str)
parser.add_argument('--save_path', default='./random_basis', type=str)
parser.add_argument('--task', default='cifar10', type=str, help='options: cifar10, cifar100, tiny')
parser.add_argument('--model', default='resnet20', type=str, help='options: lenet, alexnet, resnet20, resnet56, convnet')
device = "cuda:0"
args = parser.parse_args()



# Set up torch and cuda
dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')


#basic setups
n = args.k
max_acc = 0
test_interval = 1
epochs = args.epoch
window = args.window
log_rate = args.log_rate
batch_size = args.batch_size

    # Setup model
test_net = Siren(
        dim_in=2,
        dim_hidden=args.layer_size,
        dim_out=3,
        num_layers=args.num_layers,
        final_activation=torch.nn.Identity(),
        w0_initial=args.w0_initial,
        w0=args.w0
    ).to(device)

train_net = Siren(
        dim_in=2,
        dim_hidden=args.layer_size,
        dim_out=3,
        num_layers=args.num_layers,
        final_activation=torch.nn.Identity(),
        w0_initial=args.w0_initial,
        w0=args.w0
    ).to(device)


test_net = nn.DataParallel(test_net.cuda())
train_net = nn.DataParallel(train_net.cuda())
alpha = torch.zeros(args.k, requires_grad=True, device="cuda:0")

print(f'Image 1')

    # Load image
img = imageio.imread(f"kodak-dataset/kodim{str(1).zfill(2)}.png")
img = transforms.ToTensor()(img).float().to(device, dtype)


with torch.no_grad():
    theta = torch.cat([p.flatten() for p in train_net.parameters()])
net_optimizer = optim.SGD(train_net.parameters(), lr=1.)
lin_comb_net = torch.zeros(theta.shape).cuda()
layer_cnt = len([p for p in train_net.parameters()])
shapes = [list(p.shape) for p in train_net.parameters()]
lengths = [p.flatten().shape[0] for p in train_net.parameters()]


perm = [i for i in range(n)]
basis_net = torch.zeros(window, theta.shape[0]).cuda()
dummy_net = [torch.zeros(p.shape).cuda() for p in train_net.parameters()]
grads = torch.zeros(theta.shape, device='cuda:0')

#initializing basis networks
def fill_net(permute):
    bound = 1
    for j, p in enumerate(permute):
        torch.cuda.manual_seed_all(p + n * args.seed)
        start_ind = 0
        for i in range(layer_cnt):
            if len(shapes[i]) > 2:
                torch.nn.init.kaiming_uniform_(dummy_net[i], a=math.sqrt(5))
                basis_net[j][start_ind:start_ind + lengths[i]] = dummy_net[i].flatten()
                start_ind += lengths[i]
                bound = 1 / math.sqrt(shapes[i][1] * shapes[i][2] * shapes[i][3])
            if len(shapes[i]) == 2:
                bound = 1 / math.sqrt(shapes[i][1])
                torch.nn.init.uniform_(dummy_net[i], -bound, bound)
                basis_net[j][start_ind:start_ind + lengths[i]] = dummy_net[i].flatten()
                start_ind += lengths[i]
            if len(shapes[i]) < 2:
                torch.nn.init.uniform_(dummy_net[i], -bound, bound)
                basis_net[j][start_ind:start_ind + lengths[i]] = dummy_net[i].flatten()
                start_ind += lengths[i]
                
                
saving_path = args.save_path + '_' + args.task + '_' + args.model + '_' + str(args.k)
if args.resume:
    with torch.no_grad():
        alpha = torch.load(saving_path + '/lr.pt').cuda()
        if 'resnet' in args.model:
            means = torch.load(saving_path + '/means.pt')
            vars = torch.load(saving_path + '/vars.pt')
        ind = 0
        for p1 in train_net.modules():
            if isinstance(p1, nn.BatchNorm2d):
                leng = p1.running_var.shape[0]
                p1.running_mean.copy_(means[ind:ind + leng])
                p1.running_var.copy_(vars[ind:ind + leng])
                ind += leng
                
                
                ind += leng
else:
    with torch.no_grad():
        alpha[0] = 1.
#calculating linear combination of basis networks and alphas
def reset_lin_comb():
    global lin_comb_net
    lin_comb_net = torch.zeros(theta.shape).cuda()
    start, end = 0, window
    while start < n:
        fill_net(range(start, end))
        with torch.no_grad():
            lin_comb_net += torch.matmul(basis_net.T, alpha[start:end]).T
        start = end
        end = min(end + window, n)

reset_lin_comb()
# Dictionary to register mean values (both full precision and half precision)
results = {'fp_bpp': [], 'hp_bpp': [], 'fp_psnr': [], 'hp_psnr': []}

if args.evaluate:
    epochs = 0 

coordinates, features = util.to_coordinates_and_features(img)
coordinates, features = coordinates.to(device, dtype), features.to(device, dtype)
print_freq = 1
steps = 0  # Number of steps taken in training
best_vals = {'psnr': 0.0, 'loss': 1e8}
logs = {'psnr': [], 'loss': []}
        # Store parameters of best model (in terms of highest PSNR achieved)
best_model = OrderedDict((k, v.detach().clone()) for k, v in self.representation.state_dict().items())
        
model_size = util.model_size_in_bits(train_net) / 8000.
print(f'Model size: {model_size:.1f}kB')
fp_bpp = util.bpp(model=train_net, image=img)
print(f'Full precision bpp: {fp_bpp:.2f}') 

with tqdm.trange(5000, ncols=100) as t:
    random.shuffle(perm)
    idx = perm[:window]
    fill_net(idx)
    with torch.no_grad():
        rest_of_net = lin_comb_net - torch.matmul(basis_net.T, alpha[idx]).T
    optimizer = torch.optim.Adam([alpha], lr=args.lr)
    for i in t:
            optimizer.zero_grad()
            net_optimizer.zero_grad()
            select_subnet = torch.matmul(basis_net.T, alpha[idx]).T
            with torch.no_grad():
                start_ind = 0
                for j, p in enumerate(train_net.parameters()):
                    p.copy_((select_subnet + rest_of_net)[start_ind:start_ind + lengths[j]].view(shapes[j]))
                    start_ind += lengths[j]
            predicted = train_net(coordinates)
            loss = nn.MSE(predicted, features)
            # if i % log_rate == 0:
            #     print("Epoch:", e, "\tIteration:", i, "\tLoss:", loss.item())
            loss.backward()
            with torch.no_grad():
                start_ind = 0
                for j, p in enumerate(train_net.parameters()):
                    grads[start_ind:start_ind + lengths[j]].copy_(p.grad.flatten())
                    start_ind += lengths[j]
            if alpha.grad is None:
                alpha.grad = torch.zeros(alpha.shape, device=alpha.get_device())
            alpha.grad[idx] = torch.matmul(grads, basis_net.T)
            optimizer.step()
            
            
            
            # Calculate psnr
            psnr = get_clamped_psnr(predicted, features)

                # Print results and update logs
            log_dict = {'loss': loss.item(),
                            'psnr': psnr,
                            'best_psnr': self.best_vals['psnr']}
            t.set_postfix(**log_dict)
            for key in ['loss', 'psnr']:
                logs[key].append(log_dict[key])

                # Update best values
            if loss.item() < best_vals['loss']:
                best_vals['loss'] = loss.item()
            if psnr > best_vals['psnr']:
                best_vals['psnr'] = psnr
                    # If model achieves best PSNR seen during training, update
                    # model
                if i > int(5000 / 2.):
                    for k, v in train_net.state_dict().items():
                        best_model[k].copy_(v)


       

# Log full precision results
results['fp_bpp'].append(fp_bpp)
results['fp_psnr'].append(best_vals['psnr'])

# Save best model
torch.save(best_model, args.logdir + f'/best_model_{i}.pt')
    
# Update current model to be best model
train_net.load_state_dict(best_model)

    # Save full precision image reconstruction
with torch.no_grad():
    img_recon = train_net(coordinates).reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1)
    save_image(torch.clamp(img_recon, 0, 1).to('cpu'), args.logdir + f'/fp_reconstruction_{i}.png')    

    # Convert model and coordinates to half precision. Note that half precision
    # torch.sin is only implemented on GPU, so must use cuda
if torch.cuda.is_available():
    train_net = train_net.half().to('cuda')
    coordinates = coordinates.half().to('cuda')

        # Calculate model size in half precision
    hp_bpp = util.bpp(model=train_net, image=img)
    results['hp_bpp'].append(hp_bpp)
    print(f'Half precision bpp: {hp_bpp:.2f}')

        # Compute image reconstruction and PSNR
    with torch.no_grad():
        img_recon = train_net(coordinates).reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1).float()
        hp_psnr = util.get_clamped_psnr(img_recon, img)
        save_image(torch.clamp(img_recon, 0, 1).to('cpu'), args.logdir + f'/hp_reconstruction_{i}.png')
        print(f'Half precision psnr: {hp_psnr:.2f}')
        results['hp_psnr'].append(hp_psnr)
else:
    results['hp_bpp'].append(fp_bpp)
    results['hp_psnr'].append(0.0)


print('\n')



if args.save_model:
    torch.save(train_net.state_dict(), "final_model.pt")

