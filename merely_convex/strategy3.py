import torch
import copy
import numpy as np
import time
import csv
import argparse
import sys
sys.path.append("..")
import hypergrad as hg
import math
import higher
import numpy
import os
import scipy.io
import psutil as psutil
from torch.autograd import grad as torch_grad
from hypergrad.hypergradients import list_tensor_norm,get_outer_gradients,list_tensor_matmul,update_tensor_grads

parser = argparse.ArgumentParser(description='merely convex')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dataset', type=str, default='FashionMNIST', metavar='N')
parser.add_argument('--mode', type=str, default='ours', help='ours or IFT')
parser.add_argument('--hg_mode', type=str, default='BAMM_RHG', metavar='N',
                    help='BAMM_RHG, RHG, CG, BDA, fixed_point')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--z_loop', type=int, default=10)
parser.add_argument('--y_loop', type=int, default=1)
parser.add_argument('--x_loop', type=int, default=10000)
parser.add_argument('--z_L2_reg', type=float, default=0.01)
parser.add_argument('--y_L2_reg', type=float, default=0.001)
parser.add_argument('--y_ln_reg', type=float, default=0.1)
parser.add_argument('--reg_decay', type=bool, default=True)
parser.add_argument('--decay_rate', type=float, default=0.1)
parser.add_argument('--learn_h', type=bool, default=False)
parser.add_argument('--x_lr', type=float, default=0.005)
parser.add_argument('--y_lr', type=float, default=0.1)
parser.add_argument('--z_lr', type=float, default=0.01)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--p0', type=float, default=10)
parser.add_argument('--alpha0', type=float, default=0.1)
parser.add_argument('--beta0', type=float, default=0.1)
parser.add_argument('--eta0', type=float, default=0.5)
parser.add_argument('--mu0', type=float, default=0.9)
parser.add_argument('--x_lr_decay_rate', type=float, default=0.1)
parser.add_argument('--x_lr_decay_patience', type=int, default=1)
parser.add_argument('--pollute_rate', type=float, default=0.5)
parser.add_argument('--convex', type=bool, default=True)
parser.add_argument('--tSize', type=int, default=100)
parser.add_argument('--BDA', action='store_true',default=False)
parser.add_argument('-ew','--element_wise', action='store_true', default=False)
parser.add_argument('--linear', action='store_true', default=False)
parser.add_argument('--exp', action='store_true', default=False)
parser.add_argument('--thelr', action='store_true', default=False)# whether to use standerd strategy for update

parser.add_argument('--eta_CG', type=int, default=-1)

parser.add_argument('--c', type=float, default=2.)
parser.add_argument('--tau', type=float, default=1/40.)

parser.add_argument('--exprate', type=float, default=1.)

parser.add_argument('-ws', '--WarmStart', action='store_false', default=True)
parser.add_argument('--eta', type=float, default=1.)
args = parser.parse_args()
if not args.WarmStart or args.hg_mode.find('BAMM')!=-1:
    print('One Stage')
    # args.x_loop= args.x_loop* args.y_loop
    args.y_loop=1

print(args)

def show_memory_info(hint):
    # 获取当前进程的进程号
    pid = os.getpid()

    # psutil 是一个获取系统信息的库
    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss/1024./1024
    print(f"{hint} memory used: {memory} MB ")


def loss_L2(parameters):
    loss = 0
    for w in parameters:
        loss += torch.norm(w, 2) ** 2
    return loss


def loss_L1(parameters):
    loss = 0
    for w in parameters:
        loss += torch.norm(w, 1)
    return loss


def loss_L1(parameters):
    loss = 0
    for w in parameters:
        loss += torch.norm(w, 1)
    return loss


def accuary(out, target):
    pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    acc = pred.eq(target.view_as(pred)).sum().item() / len(target)
    return acc


def Binarization(x):
    x_bi = np.zeros_like(x)
    for i in range(x.shape[0]):
        # print(x[i])
        x_bi[i] = 1 if x[i] >= 0 else 0
    return x_bi


def cat_list_to_tensor(list_tx):
    return torch.cat([xx.view([-1]) for xx in list_tx])


class Net_x(torch.nn.Module):
    def __init__(self, tr):
        super(Net_x, self).__init__()
        self.x = torch.nn.Parameter(torch.zeros(tr.data.shape[0]).to("cuda").requires_grad_(True))

    def forward(self, y):
        # if torch.norm(torch.sigmoid(self.x), 1) > 2500:
        #     y = torch.sigmoid(self.x) / torch.norm(torch.sigmoid(self.x), 1) * 2500 * y
        # else:
        y = torch.sigmoid(self.x) * y
        y = y.mean()
        return y


def copy_parameter(y, z):
    # print(loss_L1(y.parameters()))
    # print(loss_L1(z.parameters()))
    for p, q in zip(y.parameters(), z.parameters()):
        p.data = q.clone().detach().requires_grad_()
    # print(loss_L1(y.parameters()))
    # print('-'*80)
    return y


def frnp(x):
    t = torch.from_numpy(x).cuda()
    return t


def tonp(x):
    return x.detach().cpu().numpy()


def copy_parameter(y, z):

    for p, q in zip(y.parameters(), z.parameters()):
        p.data = q.clone().detach().requires_grad_()

    return y


def copy_parameter_from_list(y, z):

    for p, q in zip(y.parameters(), z):
        p.data = q.clone().detach().requires_grad_()

    return y

gradlist=[]
xlist = []
ylist = []
zlist = []
Flist=[]
etalist=[]
xgradlist=[]
alphalist = []
timelist = []
xstar=torch.ones([args.tSize,1]).cuda()
ystar=torch.ones([args.tSize,1]).cuda()
eta=args.c*args.eta
for x0,y0 in zip([0.01],[0.01]):
    for a,b in zip([2],[2]):

        log_path = "yloop{}_tSize{}_xy{}{}_{}_BDA{}_{}_convex_{}.mat".format(args.y_loop,args.tSize,x0,y0,args.hg_mode,args.BDA,args.c, time.strftime("%Y_%m_%d_%H_%M_%S"),
                                                                   )
        with open(log_path, 'a', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow( ['y_loop{}y_lr{}x_lr{}'.format(args.y_loop,args.y_lr, args.x_lr),
                         'time', 'x','hyper_time', 'lower_time','y'])
        d = 28 ** 2
        n = 10
        z_loop = args.z_loop
        y_loop = args.y_loop
        x_loop = args.x_loop

        val_losses = []


        class ModelTensorF(torch.nn.Module):
            def __init__(self, tensor):
                super(ModelTensorF, self).__init__()
                self.T = torch.nn.Parameter(tensor)

            def forward(self,i=-1):
                if i==-1:
                    return self.T
                else:
                    return self.T[i]


        class ModelTensorf(torch.nn.Module):
            def __init__(self, tensor):
                super(ModelTensorf, self).__init__()
                self.T1 = torch.nn.Parameter(tensor)
                self.T2 = torch.nn.Parameter(tensor)

            def forward(self, i=1):
                if i == 1:
                    return self.T1
                else:
                    return self.T2

        tSize=args.tSize

        x = ModelTensorF((float(x0)*torch.ones(tSize)).cuda().requires_grad_(True))
        y = ModelTensorf((float(y0)*torch.ones(tSize)).cuda().requires_grad_(True))

        C=(float(2)*torch.ones(tSize)).cuda().requires_grad_(False)
        e=torch.ones(tSize).cuda().requires_grad_(False)


        def val_loss(params, hparams=0):

            val_loss =0.5* torch.norm(x(-1)-fmodel(2,params=params))**2 + 0.5*torch.norm(fmodel(1,params=params)-e)**2

            return val_loss

        inner_losses = []


        def train_loss(params, hparams=0):
            out=0.5*torch.norm(fmodel(1,params=params))**2-sum(x(-1)*fmodel(1,params=params))

            return out

        inner_losses = []


        def train_loss_BDA(params, hparams=0,alpha = args.alpha):

            out=(1-alpha)*train_loss(params)+alpha*val_loss(params)
            return out

        def inner_loop(hparams, params, optim, n_steps, log_interval, create_graph=False):
            params_history = [optim.get_opt_params(params)]

            for t in range(n_steps):
                params_history.append(optim(params_history[-1], hparams, create_graph=True))


            return params_history

        def inner_loop_CG(hparams, params, optim, n_steps, log_interval, create_graph=False):
            params_history = [optim.get_opt_params(params)]

            for t in range(n_steps):
                params_history=[(optim(params_history[-1], hparams))]
            return params_history


        x_opt = torch.optim.SGD(x.parameters(), lr=args.x_lr)
        acc_history = []
        clean_acc_history = []

        loss_x_l = 0
        F1_score_last = 0
        lr_decay_rate = 1
        reg_decay_rate = 1
        dc = 0
        total_time = 0
        total_hyper_time = 0
        v=-1
        # show_memory_info(1)
        for x_itr in range(x_loop):
            args.alpha=args.alpha*(x_itr+1)/(x_itr+2)
            if args.linear:
                eta = args.eta - (args.eta * (x_itr + 1) / x_loop)
            if args.exp:
                eta = eta * args.exprate
            if x_itr > 100:
                args.GN = False

            if args.thelr:
                args.alpha=args.mu0*1/(x_itr+1)**(1/args.p0)
                eta = (x_itr+1)**(-0.5 * args.tau) * args.y_lr
                args.x_lr=(x_itr+1)**(-1.5 * args.tau)*args.alpha**3*args.y_lr
                for params in x_opt.param_groups:
                    params['lr'] =  args.x_lr
                #print(x_opt.param_groups[0]['lr'])

            a= torch.any(torch.isnan(x()))
            if  torch.any(torch.isnan(x())):
                break
            x_opt.zero_grad()

            if args.hg_mode == 'CG':
                t0 = time.time()

                fmodel = higher.monkeypatch(y, torch.device("cuda"), copy_initial_weights=True,track_higher_grads=False)
                inner_opt = hg.GradientDescent(train_loss, step_size=args.y_lr)

                last_param = inner_loop_CG(x.parameters(), y.parameters(), inner_opt, y_loop, log_interval=10)
                hyper_time = time.time()

                cg_fp_map = hg.GradientDescent(loss_f=train_loss, step_size=args.y_lr)
                gts=hg.CG(last_param[0], list(x.parameters()), K=40, fp_map=cg_fp_map, outer_loss=val_loss)
            elif args.hg_mode == 'fixed_point':
                t0 = time.time()

                fmodel = higher.monkeypatch(y, torch.device("cuda"), copy_initial_weights=True,track_higher_grads=False)
                inner_opt = hg.GradientDescent(train_loss, step_size=args.y_lr)

                last_param = inner_loop_CG(x.parameters(), y.parameters(), inner_opt, y_loop, log_interval=10)
                hyper_time = time.time()

                gts=hg.fixed_point(last_param[-1], list(x.parameters()), K=40, fp_map=inner_opt,
                               outer_loss=val_loss)
            elif args.hg_mode == 'RHG':
                t0 = time.time()

                fmodel = higher.monkeypatch(y, torch.device("cuda"), copy_initial_weights=True)
                inner_opt = hg.GradientDescent(train_loss, step_size=args.y_lr)
                last_param = inner_loop(x.parameters(), y.parameters(), inner_opt, y_loop, log_interval=10)
                hyper_time = time.time()
                gts=hg.reverse(last_param[-y_loop - 1:], list(x.parameters()), [inner_opt] * y_loop, val_loss)

            elif args.hg_mode == 'BDA':
                t0 = time.time()

                fmodel = higher.monkeypatch(y, torch.device("cuda"), copy_initial_weights=True)
                inner_opt = hg.GradientDescent(train_loss_BDA, step_size=args.y_lr)
                last_param = inner_loop(x.parameters(), y.parameters(), inner_opt, y_loop, log_interval=10)
                hyper_time = time.time()
                gts=hg.reverse(last_param[-y_loop - 1:], list(x.parameters()), [inner_opt] * y_loop, val_loss)

            elif args.hg_mode == 'BAMM_CG':
                t0 = time.time()
                fmodel = higher.monkeypatch(y, torch.device("cuda"), copy_initial_weights=True)
                inner_opt = hg.GradientDescent(train_loss_BDA if args.BDA else train_loss, step_size=args.y_lr)
                new_time = time.time()
                last_param = inner_loop(x.parameters(), y.parameters(), inner_opt, 1, log_interval=10)
                y_time = time.time() - new_time
                params = [w.detach().requires_grad_(True) for w in list(y.parameters())]  # y
                o_loss = val_loss(params, list(x.parameters()))  # F
                grad_outer_w, grad_outer_hparams = get_outer_gradients(o_loss, params,
                                                                       list(x.parameters()))  # dy F,dx F
                w_mapped = inner_opt(params, list(x.parameters()), only_grad=True)  # dy f
                prepare_time = time.time() - t0 - y_time
                hyper_time = time.time()
                vs = [torch.zeros_like(w) for w in params] if v == -1 else v

                vsp = torch_grad(w_mapped, params, grad_outputs=vs, retain_graph=True,
                                 allow_unused=True)  # dy (dy f) v=d2y f v
                tem = [v - gow for v, gow in zip(vsp, grad_outer_w)]

                ita_u = list_tensor_norm(tem) ** 2
                grad_tem = torch_grad(w_mapped, params, grad_outputs=tem, retain_graph=True,
                                      allow_unused=True)  # dy (dy f) v=d2y f v

                ita_l = list_tensor_matmul(tem, grad_tem, trans=1)
                # print(ita_u,ita_l)
                ita = ita_u / (ita_l + 1e-12)
                vs = [v0 - ita * v + ita * gow for v0, v, gow in zip(vs, vsp, grad_outer_w)]  # (I-ita*d2yf)v+ita*dy F)

                v_time = time.time() - hyper_time
                new_time = time.time()
                grads = torch_grad(w_mapped, list(x.parameters()),
                                   grad_outputs=[torch.zeros_like(w) for w in params] if v == -1 else v,
                                   allow_unused=True)  # dx (dy f) v

                grads = [-g + v if g is not None else v for g, v in zip(grads, grad_outer_hparams)]

                # if set_grad:
                update_tensor_grads(list(x.parameters()), grads)

                x_time = time.time() - new_time
                v = vs
                gts = grads
                eta = ita
            else:
                print('NO hypergradient!')
            xgrad=[x0g.grad for x0g in x.parameters()]
            # print(eta)

            x_opt.step()
            # if args.hg_mode != 'BDA':
            copy_parameter_from_list(y, last_param[-1])
            if 'BAMM' in args.hg_mode:

                time_list=[y_time,prepare_time+x_time,prepare_time+v_time]
                outer_time=max(time_list)
                step_time = outer_time
                total_hyper_time += prepare_time+x_time+v_time
                total_time += outer_time
            else:

                step_time=time.time() - t0
                total_hyper_time += time.time() - hyper_time
                total_time += time.time() - t0

            if x_itr % 10 == 0:

                xgradlist.append(copy.deepcopy([x0.detach().cpu().numpy() for x0 in xgrad]))

                with torch.no_grad():
                    with torch.no_grad():

                        print(
                            'x_itr={},xdist={:.6f},ydist={:.6f},zdist={:.6f},xgrad={:.6f}, total_hyper_time={:.6f}, total_time={:.6f}'.format(
                                x_itr, torch.norm((x() - xstar) / xstar).detach().cpu().numpy(),
                                (torch.norm((y(1) - ystar)) / torch.norm(ystar)).detach().cpu().numpy(),
                                (torch.norm((y(2) - ystar)) / torch.norm(ystar)).detach().cpu().numpy(),
                                xgradlist[-1][0][0],
                                total_hyper_time, total_time))
                        xlist.append(copy.deepcopy(x().detach().cpu().numpy()))
                        ylist.append(y().detach().cpu().numpy())
                        zlist.append(copy.deepcopy(y(2).detach().cpu().numpy()))

                        timelist.append(total_time)
                        Flist.append(val_loss(last_param[-1]).detach().cpu().numpy())
                        alphalist.append(args.alpha)


            if len(timelist)>600:#timelist[-1]>15:# loss_L2(xgrad) < 1e-6:
                print(
                    'x_itr={},xdist={:.6f},ydist={:.6f},zdist={:.6f},xgrad={:.6f}, total_hyper_time={:.6f}, total_time={:.6f}'.format(
                        x_itr, torch.norm((x() - xstar) / xstar).detach().cpu().numpy(),
                        (torch.norm((y(1) - ystar)) /torch.norm (ystar)).detach().cpu().numpy(),
                        (torch.norm((y(2) - ystar)) / torch.norm(ystar)).detach().cpu().numpy(),

                        xgradlist[-1][0][0],
                        total_hyper_time, total_time))
                xgradlist.append(copy.deepcopy([x0.detach().cpu().numpy() for x0 in xgrad]))

                # print(torch.autograd.grad(train_loss(last_param[-1]),y.parameters()))
                with torch.no_grad():
                    with torch.no_grad():

                        xlist.append(copy.deepcopy(x().detach().cpu().numpy()))
                        ylist.append(copy.deepcopy(y(1).detach().cpu().numpy()))
                        zlist.append(copy.deepcopy(y(2).detach().cpu().numpy()))
                        alphalist.append(args.alpha)

                        timelist.append(total_time)
                        Flist.append(val_loss(last_param[-1]).detach().cpu().numpy())
                break

scipy.io.savemat(log_path, mdict={'x': xlist, 'y': ylist, 'z': zlist, 'time': timelist, 'n': args.tSize,
                                                  'xstar': xstar.cpu().numpy(), 'F': Flist, 'xgrad': xgradlist,
                                                  'eta': etalist,'alpha':alphalist, 'Memory': torch.cuda.max_memory_allocated()})