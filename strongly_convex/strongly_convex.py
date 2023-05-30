import torch
import copy
import numpy as np
import time
import csv
import argparse
import hypergrad as hg
import math
import higher
import numpy
import os
import scipy.io


import psutil as psutil
from torch.autograd import grad as torch_grad
from hypergrad.hypergradients import list_tensor_norm,get_outer_gradients,list_tensor_matmul,update_tensor_grads

parser = argparse.ArgumentParser(description='strongly_convex')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dataset', type=str, default='FashionMNIST', metavar='N')
parser.add_argument('--mode', type=str, default='ours', help='ours or IFT')
parser.add_argument('--hg_mode', type=str, default='RHG', metavar='N',
                    help='hypergradient RHG or CG or fixed_point')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--z_loop', type=int, default=10)
parser.add_argument('--y_lin_loop', type=int, default=50)
parser.add_argument('-le','--lin_error', type=float, default=-1.)
parser.add_argument('--y_loop', type=int, default=100)
parser.add_argument('--x_loop', type=int, default=5000)
parser.add_argument('--z_L2_reg', type=float, default=0.01)
parser.add_argument('--y_L2_reg', type=float, default=0.001)
parser.add_argument('--y_ln_reg', type=float, default=0.1)
parser.add_argument('--reg_decay', type=bool, default=True)
parser.add_argument('--decay_rate', type=float, default=0.1)
parser.add_argument('--learn_h', type=bool, default=False)
parser.add_argument('--x_lr', type=float, default=0.01)
parser.add_argument('--y_lr', type=float, default=0.2)
parser.add_argument('--z_lr', type=float, default=0.01)
parser.add_argument('--x_lr_decay_rate', type=float, default=0.1)
parser.add_argument('--x_lr_decay_patience', type=int, default=1)
parser.add_argument('--pollute_rate', type=float, default=0.5)
parser.add_argument('--convex', type=bool, default=True)
parser.add_argument('--tSize', type=int, default=1)
parser.add_argument('--xSize', type=int, default=100)
parser.add_argument('--ySize', type=int, default=100)
parser.add_argument('--log', type=int, default=5)#10
parser.add_argument('--idt', action='store_false', default=True)
parser.add_argument('--BDA', action='store_true', default=False)
parser.add_argument('--GN', action='store_true', default=False)
parser.add_argument('--linear', action='store_true', default=False)
parser.add_argument('--exp', action='store_true', default=False)
parser.add_argument('--back', action='store_true', default=False)
parser.add_argument('-ew','--element_wise', action='store_true', default=False)
parser.add_argument('--tau', type=float, default=1/1000.)
parser.add_argument('--eta_CG', type=int, default=-1)
parser.add_argument('--thelr', action='store_true', default=False)
parser.add_argument('--c', type=float, default=2.0)#0.125

parser.add_argument('--exprate', type=float, default=0.99)
parser.add_argument('--eta0', type=float, default=0.125)#0.5
parser.add_argument('-ws', '--WarmStart', action='store_false', default=True)
parser.add_argument('--eta', type=float, default=1.)
args = parser.parse_args()
if not args.WarmStart or args.hg_mode.find('Darts') != -1:
    print('One Stage')
    args.x_loop = args.x_loop * args.y_loop
    args.y_loop = 1

args.tSize=max(args.xSize,args.ySize)
if_cuda=False

print(args)


def show_memory_info(hint):
    # 获取当前进程的进程号
    pid = os.getpid()

    # psutil 是一个获取系统信息的库
    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss / 1024. / 1024
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


def positive_matrix(m):
    randt = torch.rand(m) + 1
    matrix0 = torch.diag(randt)
    invmatrix0 = torch.diag(1 / randt)
    Q = torch.rand(m, m)
    Q, R = torch.qr(Q)
    matrix = torch.mm(torch.mm(Q.t(), matrix0), Q)
    invmatrix = torch.mm(torch.mm(Q.t(), invmatrix0), Q)
    return matrix, invmatrix


def semipositive_matrix(m):
    randt = torch.rand(m)
    matrix0 = torch.diag(randt)
    invmatrix0 = torch.diag(1 / randt)
    Q = torch.rand(m, m)
    Q, R = torch.qr(Q)
    matrix = torch.mm(torch.mm(Q.t(), matrix0), Q)
    invmatrix = torch.mm(torch.mm(Q.t(), invmatrix0), Q)
    return matrix, invmatrix


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
    # print(loss_L1(y.parameters()))
    # print(loss_L1(z.parameters()))
    for p, q in zip(y.parameters(), z.parameters()):
        p.data = q.clone().detach().requires_grad_()
    # print(loss_L1(y.parameters()))
    # print('-'*80)
    return y


def copy_parameter_from_list(y, z):
    # print(loss_L1(y.parameters()))
    # print(loss_L1(z.parameters()))
    for p, q in zip(y.parameters(), z):
        p.data = q.clone().detach().requires_grad_()
    # print(loss_L1(y.parameters()))
    # print('-'*80)
    return y


idt = args.idt
if idt:

    A= 1
    B = 1
    D = 1
    invA = A

    xh = torch.ones([args.xSize, 1]) * 1
    ystar = 1
    BinvATBT = 1
    inv_BinvATBT_mD = 0.5
    xstar =0.5*xh
else:
    A, invA = positive_matrix(args.ySize)
    B = torch.rand([args.xSize, args.ySize])
    D, invD = semipositive_matrix(args.xSize)
    xh = torch.rand([args.xSize, 1]) * 1
    ystar = torch.mm(invA, B.t())
    BinvATBT = torch.mm(torch.mm(B, invA.t()), B.t())
    inv_BinvATBT_mD = torch.inverse(BinvATBT + D)
    xstar = torch.mm(inv_BinvATBT_mD, torch.mm(D, xh))

# x0=float(args.x0)
# y0=float(args.y0)
gradlist = []
xlist = []
ylist = []
vlist = []
Flist=[]
etalist=[]
xgradlist=[]
timelist = []

for x0, y0 in zip([2], [2]):
    for a, b in zip([2], [2]):

        log_path = r"yloop{}_tSize{}_xy{}{}_{}_BDA{}_ws{}_eta{}_exp{}_c{}_etaCG{}_ew{}_le{}._convex_{}.mat".format(args.y_loop, args.tSize,
                                                                                                x0, y0, args.hg_mode,
                                                                                                args.BDA,
                                                                                                args.WarmStart,
                                                                                                args.eta, args.exprate,
                                                                                                args.c,args.eta_CG,args.element_wise,args.lin_error, time.strftime(
                "%Y_%m_%d_%H_%M_%S"),
                                                                                                )
        # with open(log_path, 'a', encoding='utf-8') as f:
        #     csv_writer = csv.writer(f)
        #     # csv_writer.writerow([args])
        #     csv_writer.writerow( ['y_loop{}y_lr{}x_lr{}'.format(args.y_loop,args.y_lr, args.x_lr),
        #                  'time', 'x','hyper_time', 'lower_time','y'])
        d = 28 ** 2
        n = 10
        z_loop = args.z_loop
        y_loop = args.y_loop
        x_loop = args.x_loop

        val_losses = []


        class ModelTensorF(torch.nn.Module):
            def __init__(self, tensor):
                super(ModelTensorF, self).__init__()
                self.p = torch.nn.Parameter(tensor)

            def forward(self):
                return self.p

            def t(self):
                return self.p.t()


        class ModelTensorf(torch.nn.Module):
            def __init__(self, tensor):
                super(ModelTensorf, self).__init__()
                self.p = torch.nn.Parameter(tensor)

            def forward(self):
                return self.p

            def t(self):
                return self.p.t()


        tSize = args.tSize

        Fmin = 0
        xmin = []
        xmind = 0
        for i in range(tSize):
            Fmin = Fmin + (-np.pi / 4 / (i + 1) - a) ** 2 + (-np.pi / 4 / (i + 1) - b) ** 2
            xmin.append(-np.pi / 4 / (i + 1))
            xmind = xmind + (-np.pi / 4 / (i + 1)) ** 2

        x = ModelTensorF(float(x0) * torch.ones([args.xSize, 1])).requires_grad_(True)
        y = ModelTensorF(float(y0) * torch.ones([args.ySize, 1])).requires_grad_(True)


        C = (float(2) * torch.ones(tSize)).cuda().requires_grad_(False)
        e = torch.ones(tSize).requires_grad_(False)


        def val_loss(params, hparams=0):
            xmh = x() - xh
            if args.idt:
                if args.element_wise:
                    val_loss=0
                    for i in range(args.xSize):
                        val_loss=val_loss+0.5 *fmodel(params=params)[i]*fmodel(params=params)[i] + 0.5 * xmh[i]*xmh[i]

                else:
                    val_loss = 0.5 * torch.mm(fmodel(params=params).t(),fmodel(params=params)) + 0.5 * torch.mm(xmh.t(),  xmh)

            else:
                val_loss = 0.5 * torch.mm(torch.mm(fmodel(params=params).t(), A), fmodel(params=params)) + 0.5 * torch.mm(
                torch.mm(xmh.t(), D), xmh)

            return val_loss


        inner_losses = []


        def train_loss(params, hparams=0):
            if args.idt:
                if args.element_wise:
                    out = 0
                    for i in range(args.xSize):
                        out =out+ 0.5 * fmodel(params=params)[i]* fmodel(params=params)[i] -x()[i]*fmodel(params=params)[i]
                else:
                    out = 0.5 * torch.mm(fmodel(params=params).t(), fmodel(params=params)) -torch.mm(x.t(),  fmodel(params=params))
            else:

                out = 0.5 * torch.mm(torch.mm(fmodel(params=params).t(), A), fmodel(params=params)) - torch.mm(
                torch.mm(x.t(), B), fmodel(params=params))

            return out


        # def val_loss_BDA(params,alpha=0.5):
        #     BDA_loss=(1-alpha)*train_loss(params)+alpha*val_loss(params)
        #     return val_loss

        inner_losses = []


        def train_loss_BDA(params, hparams=0, alpha=0.1):
            out = (1 - alpha) * train_loss(params) + alpha * val_loss(params)
            return out


        def inner_loop(hparams, params, optim, n_steps, log_interval, create_graph=False):
            params_history = [optim.get_opt_params(params)]

            for t in range(n_steps):
                params_history.append(optim(params_history[-1], hparams, create_graph=True))

                # if log_interval and (t % log_interval == 0 or t == n_steps-1):
                #     print('t={}, Loss: {:.6f}'.format(t, optim.curr_loss.item()))

            return params_history


        def inner_loop_CG(hparams, params, optim, n_steps, log_interval, create_graph=False):
            params_history = [optim.get_opt_params(params)]

            for t in range(n_steps):
                params_history = [(optim(params_history[-1], hparams))]

                # if log_interval and (t % log_interval == 0 or t == n_steps-1):
                #     print('t={}, Loss: {:.6f}'.format(t, optim.curr_loss.item()))

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
        v = -1
        eta = args.eta * args.c
        # show_memory_info(1)
        # A=A.cuda()
        # B=B.cuda()
        # D=D.cuda()
        if if_cuda:
            x=x.cuda()
            y=y.cuda()
            e=e.cuda()
            xh=xh.cuda()
            xstar=xstar.cuda()
        # ystar=ystar.cuda()

        for x_itr in range(x_loop):
            x_opt.zero_grad()
            if args.linear:
                eta = args.eta - (args.eta * (x_itr + 1) / x_loop)
            if args.exp:
                eta = eta * args.exprate
            if x_itr > 100:
                args.GN = False

            if args.thelr:
                #args.alpha = args.mu0*1/(x_itr+1)**(1/args.p)
                x_lr=args.x_lr*(x_itr+1)**(-args.tau)*args.y_lr
                # args.y_lr=1/(x_itr+1)**0.01
                eta=args.eta0*(x_itr+1)**(-0.5*args.tau)*args.y_lr

                for params in x_opt.param_groups:
                    params['lr'] =  x_lr

            # if x_itr == 0:
            #     with torch.no_grad():
            #         # x_opt_l.step(F1_score)
            #         # y_opt_l.step(acc)
            #         # z_opt_l.step(acc)
            #         fmodel = higher.monkeypatch(y, torch.device("cuda"), copy_initial_weights=True)
            #         print(
            #             'x_itr={},xdist={:.6f},ydist={:.6f}, total_hyper_time={:.6f}, total_time={:.6f}'.format(
            #                 x_itr, torch.norm((x() - xstar) / xstar).detach().cpu().numpy(),
            #                 torch.norm((y() - torch.mm(ystar, x())) / torch.mm(ystar, x())).detach().cpu().numpy() if not args.idt else torch.norm((y() -ystar*x()) / (ystar* x())).detach().cpu().numpy(),
            #                 total_hyper_time, total_time))
            #         xlist.append(copy.deepcopy(x().detach().cpu().numpy()))
            #         ylist.append(y().detach().cpu().numpy())
            #         etalist.append(eta.detach().cpu().numpy() if torch.is_tensor(eta) else eta)
            #         if v == -1:
            #             vlist.append(v)
            #         else:
            #             vlist.append([v0.detach().cpu().numpy() for v0 in v])
            #         timelist.append(total_time)
            #         Flist.append(val_loss(list(y.parameters())).detach().cpu().numpy())


            if args.hg_mode == 'CG':
                t0 = time.time()

                fmodel = higher.monkeypatch(y, torch.device("cuda"), copy_initial_weights=True,
                                            track_higher_grads=False)
                inner_opt = hg.GradientDescent(train_loss, step_size=args.y_lr)

                last_param = inner_loop_CG(x.parameters(), y.parameters(), inner_opt, y_loop, log_interval=10)
                hyper_time = time.time()

                # This is the approximation used in the paper CG stands for conjugate gradient
                cg_fp_map = hg.GradientDescent(loss_f=train_loss, step_size=args.y_lr)
                gts = hg.CG(last_param[0], list(x.parameters()), K=100 if args.lin_error>0 else 40, fp_map=cg_fp_map,
                            outer_loss=val_loss,tol=args.lin_error)
            elif args.hg_mode == 'fixed_point':
                t0 = time.time()

                fmodel = higher.monkeypatch(y, torch.device("cuda"), copy_initial_weights=True,
                                            track_higher_grads=False)
                inner_opt = hg.GradientDescent(train_loss, step_size=args.y_lr)

                last_param = inner_loop_CG(x.parameters(), y.parameters(), inner_opt, y_loop, log_interval=10)
                hyper_time = time.time()

                gts = hg.fixed_point(last_param[-1], list(x.parameters()), K= args.y_lin_loop,
                                     fp_map=inner_opt,
                                     outer_loss=val_loss,tol=args.lin_error)
            elif args.hg_mode == 'RHG':
                t0 = time.time()

                fmodel = higher.monkeypatch(y, torch.device("cuda"), copy_initial_weights=True)
                inner_opt = hg.GradientDescent(train_loss, step_size=args.y_lr)
                last_param = inner_loop(x.parameters(), y.parameters(), inner_opt, y_loop, log_interval=10)
                hyper_time = time.time()
                gts = hg.reverse(last_param[-y_loop - 1:], list(x.parameters()), [inner_opt] * y_loop, val_loss)
            elif args.hg_mode == 'TRHG':
                t0 = time.time()

                fmodel = higher.monkeypatch(y, torch.device("cuda"), copy_initial_weights=True)
                inner_opt = hg.GradientDescent(train_loss, step_size=args.y_lr)
                last_param = inner_loop(x.parameters(), y.parameters(), inner_opt, y_loop, log_interval=10)
                hyper_time = time.time()
                gts = hg.reverse(last_param[int(-y_loop / 2) - 1:], list(x.parameters()), [inner_opt] * y_loop,
                                 val_loss)
            elif args.hg_mode == 'IAPTT':
                t0 = time.time()

                fmodel = higher.monkeypatch(y, torch.device("cuda"), copy_initial_weights=True)
                inner_opt = hg.GradientDescent(train_loss, step_size=args.y_lr)
                last_param = inner_loop(x.parameters(), y.parameters(), inner_opt, y_loop, log_interval=10)
                hyper_time = time.time()
                Flist = [val_loss(param).detach().cpu().numpy() for param in last_param]
                maxk = np.argmax(Flist)
                gts = hg.reverse(last_param[-maxk - 1:], list(x.parameters()), [inner_opt] * y_loop, val_loss)
            elif args.hg_mode == 'BDA':
                t0 = time.time()

                fmodel = higher.monkeypatch(y, torch.device("cuda"), copy_initial_weights=True)
                inner_opt = hg.GradientDescent(train_loss_BDA, step_size=args.y_lr)
                last_param = inner_loop(x.parameters(), y.parameters(), inner_opt, y_loop, log_interval=10)
                hyper_time = time.time()
                gts = hg.reverse(last_param[-y_loop - 1:], list(x.parameters()), [inner_opt] * y_loop, val_loss)
            elif args.hg_mode == 'Darts_W_RHG' and (args.eta_CG==-1 or x_itr%args.eta_CG!=0):
                t0 = time.time()

                fmodel = higher.monkeypatch(y, torch.device("cuda"), copy_initial_weights=True)
                inner_opt = hg.GradientDescent(train_loss_BDA if args.BDA else train_loss, step_size=args.y_lr)
                last_param = inner_loop(x.parameters(), y.parameters(), inner_opt, 1, log_interval=10)
                hyper_time = time.time()
                gts, v = hg.Darts_W_RHG(last_param[-1], list(x.parameters()), K=40, fp_map=inner_opt,
                                        outer_loss=val_loss, v0=v, ita=eta)
            elif args.hg_mode == 'Darts_W_RHG_mp' and (args.eta_CG==-1 or x_itr%args.eta_CG!=0):
                t0 = time.time()
                fmodel = higher.monkeypatch(y, torch.device("cuda"), copy_initial_weights=True)
                inner_opt = hg.GradientDescent(train_loss_BDA if args.BDA else train_loss, step_size=args.y_lr)
                new_time = time.time()
                last_param = inner_loop(x.parameters(), y.parameters(), inner_opt, 1, log_interval=10)
                y_time = time.time() - new_time
                params = [w.detach().requires_grad_(True) for w in list(y.parameters())]  # y
                o_loss = val_loss(params, list(x.parameters()))  # F
                grad_outer_w, grad_outer_hparams = get_outer_gradients(o_loss, params, list(x.parameters()))  # dy F,dx F
                w_mapped = inner_opt(params, list(x.parameters()), only_grad=True)  # dy f
                prepare_time = time.time() - t0 - y_time
                hyper_time = time.time()
                vs = [torch.zeros_like(w) for w in params] if v == -1 else v
                vsp = torch_grad(w_mapped, params, grad_outputs=vs, retain_graph=True,
                                 allow_unused=True)  # dy (dy f) v=d2y f v
                vs = [v0 - eta * (v if v is not None else 0) + eta * (gow if gow is not None else 0) for v0, v, gow in
                      zip(vs, vsp, grad_outer_w)]  # (I-ita*d2yf)v+ita*dy F)
                v_time = time.time() - hyper_time
                new_time = time.time()
                grads = torch_grad(w_mapped, list(x.parameters()),
                                   grad_outputs=[torch.zeros_like(w) for w in params] if v == -1 else v,
                                   allow_unused=True)  # dx (dy f) v
                grads = [-g + v if g is not None else v for g, v in zip(grads, grad_outer_hparams)]

                update_tensor_grads(list(x.parameters()), grads)
                x_time = time.time() - new_time
                v = vs
                gts = grads


            elif args.hg_mode == 'Darts_W_CG' or (args.eta_CG!=-1 and x_itr%args.eta_CG==0 and 'mp' not in args.hg_mode):
                t0 = time.time()
                # print ('CG eta')
                fmodel = higher.monkeypatch(y, torch.device("cuda"), copy_initial_weights=True)
                inner_opt = hg.GradientDescent(train_loss_BDA if args.BDA else train_loss, step_size=args.y_lr)
                last_param = inner_loop(x.parameters(), y.parameters(), inner_opt, 1, log_interval=10)
                hyper_time = time.time()
                gts, v, eta = hg.Darts_W_CG(last_param[-1], list(x.parameters()), K=40, fp_map=inner_opt,
                                            outer_loss=val_loss, v0=v)
            elif args.hg_mode == 'Darts_W_CG_mp' or (args.eta_CG!=-1 and x_itr%args.eta_CG==0 and 'mp' in args.hg_mode):
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

            x_opt.step()
            # if args.hg_mode != 'BDA':

            copy_parameter_from_list(y, last_param[-1])

            if 'mp' in args.hg_mode:
                time_list=[y_time,prepare_time+x_time,prepare_time+v_time]
                outer_time=max(time_list)
                step_time = outer_time
                total_hyper_time += prepare_time+x_time+v_time
                total_time += outer_time
            else:
                step_time = time.time() - t0

                total_hyper_time += time.time() - hyper_time
                total_time += time.time() - t0


            # if not torch.is_tensor(v):
            #     v=torch.tensor(v).cuda()
            # y_history=[yh[0].item() for yh in last_param]

            if x_itr % args.log == 0:
                # print(abs(val_loss(y.T1.data,1)+1))
                # print(total_time/(x_itr+1))
                # print(val_loss(last_param[-1]))
                xgradlist.append(copy.deepcopy([x0.detach().cpu().numpy() for x0 in xgrad]))


                # print(torch.autograd.grad(train_loss(last_param[-1]),y.parameters()))
                with torch.no_grad():
                    with torch.no_grad():
                        # x_opt_l.step(F1_score)
                        # y_opt_l.step(acc)
                        # z_opt_l.step(acc)

                        print(
                            'x_itr={},xdist={:.6f},ydist={:.6f},xgrad={:.6f}, total_hyper_time={:.6f}, total_time={:.6f}'.format(
                                x_itr, torch.norm((x() - xstar) / xstar).detach().cpu().numpy(),
                                torch.norm((y() - torch.mm(ystar, x())) / torch.mm(ystar, x())).detach().cpu().numpy() if not args.idt else torch.norm((y() -ystar*x()) / (ystar* x())).detach().cpu().numpy(),xgradlist[-1][0][0][0],
                                total_hyper_time, total_time))
                        xlist.append(copy.deepcopy(x().detach().cpu().numpy()))
                        ylist.append(y().detach().cpu().numpy())
                        etalist.append(eta.detach().cpu().numpy() if torch.is_tensor(eta) else eta)
                        if v == -1:
                            vlist.append(v)
                        else:
                            vlist.append([v0.detach().cpu().numpy() for v0 in v])
                        timelist.append(total_time)
                        Flist.append(val_loss(last_param[-1]).detach().cpu().numpy())
                        if len(xlist) > 1:
                            # print(np.linalg.norm(xlist[-1]-xlist[-2])/np.linalg.norm(xlist[-2]))
                            # print(np.linalg.norm(xlist[-1]-xlist[-2]),eta)
                            print(loss_L2(xgrad))
            if len(xlist)>1:
                # if (args.hg_mode == 'RHG'):
                #     if len(timelist) > 200 or args.element_wise:  # or args.element_wise:#timelist[-1]>20loss_L2(xgrad)<1e-8:#loss_L2(xgrad)<1e-4:#timelist[-1]>6:#loss_L2(xgrad)<1e-8:#np.linalg.norm(xlist[-1]-xlist[-2])/np.linalg.norm(xlist[-2])<1e-8:
                #         print(
                #             'x_itr={},xdist={:.6f},ydist={:.6f},xgrad={:.6f}, total_hyper_time={:.6f}, total_time={:.6f}'.format(
                #                 x_itr, torch.norm((x() - xstar) / xstar).detach().cpu().numpy(),
                #                 torch.norm((y() - torch.mm(ystar, x())) / torch.mm(ystar,
                #                                                                    x())).detach().cpu().numpy() if not args.idt else torch.norm(
                #                     (y() - ystar * x()) / (ystar * x())).detach().cpu().numpy(),
                #                 xgradlist[-1][0][0][0],
                #                 total_hyper_time, total_time))
                #         xgradlist.append(copy.deepcopy([x0.detach().cpu().numpy() for x0 in xgrad]))
                #
                #         # print(torch.autograd.grad(train_loss(last_param[-1]),y.parameters()))
                #         with torch.no_grad():
                #             with torch.no_grad():
                #                 # x_opt_l.step(F1_score)
                #                 # y_opt_l.step(acc)
                #                 # z_opt_l.step(acc)
                #                 # print(xgradlist[-1][0][0][0])
                #                 # print(
                #                 #     'x_itr={},xdist={:.6f},ydist={:.6f},xgrad={:.6f}, total_hyper_time={:.6f}, total_time={:.6f}'.format(
                #                 #         x_itr, torch.norm((x() - xstar) / xstar).detach().cpu().numpy(),
                #                 #         torch.norm((y() - torch.mm(ystar, x())) / torch.mm(ystar, x())).detach().cpu().numpy(),xgradlist[-1][0][0][0],
                #                 #         total_hyper_time, total_time))
                #                 xlist.append(copy.deepcopy(x().detach().cpu().numpy()))
                #                 ylist.append(y().detach().cpu().numpy())
                #                 etalist.append(eta.detach().cpu().numpy() if torch.is_tensor(eta) else eta)
                #                 if v == -1:
                #                     vlist.append(v)
                #                 else:
                #                     vlist.append([v0.detach().cpu().numpy() for v0 in v])
                #                 timelist.append(total_time)
                #                 Flist.append(val_loss(last_param[-1]).detach().cpu().numpy())
                #         break
                # elif (args.hg_mode == 'CG') or (args.hg_mode == 'fixed_point'):
                #     if len(timelist) > 100 or args.element_wise:  # or args.element_wise:#timelist[-1]>20loss_L2(xgrad)<1e-8:#loss_L2(xgrad)<1e-4:#timelist[-1]>6:#loss_L2(xgrad)<1e-8:#np.linalg.norm(xlist[-1]-xlist[-2])/np.linalg.norm(xlist[-2])<1e-8:
                #         print(
                #             'x_itr={},xdist={:.6f},ydist={:.6f},xgrad={:.6f}, total_hyper_time={:.6f}, total_time={:.6f}'.format(
                #                 x_itr, torch.norm((x() - xstar) / xstar).detach().cpu().numpy(),
                #                 torch.norm((y() - torch.mm(ystar, x())) / torch.mm(ystar,
                #                                                                    x())).detach().cpu().numpy() if not args.idt else torch.norm(
                #                     (y() - ystar * x()) / (ystar * x())).detach().cpu().numpy(),
                #                 xgradlist[-1][0][0][0],
                #                 total_hyper_time, total_time))
                #         xgradlist.append(copy.deepcopy([x0.detach().cpu().numpy() for x0 in xgrad]))
                #
                #         # print(torch.autograd.grad(train_loss(last_param[-1]),y.parameters()))
                #         with torch.no_grad():
                #             with torch.no_grad():
                #                 # x_opt_l.step(F1_score)
                #                 # y_opt_l.step(acc)
                #                 # z_opt_l.step(acc)
                #                 # print(xgradlist[-1][0][0][0])
                #                 # print(
                #                 #     'x_itr={},xdist={:.6f},ydist={:.6f},xgrad={:.6f}, total_hyper_time={:.6f}, total_time={:.6f}'.format(
                #                 #         x_itr, torch.norm((x() - xstar) / xstar).detach().cpu().numpy(),
                #                 #         torch.norm((y() - torch.mm(ystar, x())) / torch.mm(ystar, x())).detach().cpu().numpy(),xgradlist[-1][0][0][0],
                #                 #         total_hyper_time, total_time))
                #                 xlist.append(copy.deepcopy(x().detach().cpu().numpy()))
                #                 ylist.append(y().detach().cpu().numpy())
                #                 etalist.append(eta.detach().cpu().numpy() if torch.is_tensor(eta) else eta)
                #                 if v == -1:
                #                     vlist.append(v)
                #                 else:
                #                     vlist.append([v0.detach().cpu().numpy() for v0 in v])
                #                 timelist.append(total_time)
                #                 Flist.append(val_loss(last_param[-1]).detach().cpu().numpy())
                #         break
                # else:
                if loss_L2(xgrad)<1e-5:#timelist[-1]>30 or args.element_wise:# or args.element_wise:#timelist[-1]>20loss_L2(xgrad)<1e-8:#loss_L2(xgrad)<1e-4:#timelist[-1]>6:#loss_L2(xgrad)<1e-8:#np.linalg.norm(xlist[-1]-xlist[-2])/np.linalg.norm(xlist[-2])<1e-8:
                    print(
                        'x_itr={},xdist={:.6f},ydist={:.6f},xgrad={:.6f}, total_hyper_time={:.6f}, total_time={:.6f}'.format(
                            x_itr, torch.norm((x() - xstar) / xstar).detach().cpu().numpy(),
                            torch.norm((y() - torch.mm(ystar, x())) / torch.mm(ystar, x())).detach().cpu().numpy() if not args.idt else torch.norm((y() -ystar*x()) / (ystar* x())).detach().cpu().numpy(),
                            xgradlist[-1][0][0][0],
                            total_hyper_time, total_time))
                    xgradlist.append(copy.deepcopy([x0.detach().cpu().numpy() for x0 in xgrad]))


                    # print(torch.autograd.grad(train_loss(last_param[-1]),y.parameters()))
                    with torch.no_grad():
                        with torch.no_grad():
                            # x_opt_l.step(F1_score)
                            # y_opt_l.step(acc)
                            # z_opt_l.step(acc)
                            # print(xgradlist[-1][0][0][0])
                            # print(
                            #     'x_itr={},xdist={:.6f},ydist={:.6f},xgrad={:.6f}, total_hyper_time={:.6f}, total_time={:.6f}'.format(
                            #         x_itr, torch.norm((x() - xstar) / xstar).detach().cpu().numpy(),
                            #         torch.norm((y() - torch.mm(ystar, x())) / torch.mm(ystar, x())).detach().cpu().numpy(),xgradlist[-1][0][0][0],
                            #         total_hyper_time, total_time))
                            xlist.append(copy.deepcopy(x().detach().cpu().numpy()))
                            ylist.append(y().detach().cpu().numpy())
                            etalist.append(eta.detach().cpu().numpy() if torch.is_tensor(eta) else eta)
                            if v == -1:
                                vlist.append(v)
                            else:
                                vlist.append([v0.detach().cpu().numpy() for v0 in v])
                            timelist.append(total_time)
                            Flist.append(val_loss(last_param[-1]).detach().cpu().numpy())
                    break
            #
            #             # with torch.no_grad():
            #             #     out = y(test.data)
            #     print(100*accuary(out,test.clean_target))
if args.idt:
    scipy.io.savemat(log_path, mdict={'x': xlist, 'y': ylist, 'v': vlist, 'time': timelist,'n':args.xSize,
                                   'xh': xh.cpu().numpy(),
                                  'xstar': xstar.cpu().numpy(),'F':Flist,'xgrad':xgradlist,'eta':etalist,'Memory':torch.cuda.max_memory_allocated()})
else:
    scipy.io.savemat(log_path, mdict={'x': xlist, 'y': ylist, 'v': vlist, 'time': timelist, 'A': A.cpu().numpy(),
                                  'B': B.cpu().numpy(), 'D': D.cpu().numpy(), 'xh': xh.cpu().numpy(),
                                  'xstar': xstar.cpu().numpy(),'F':Flist,'xgrad':xgradlist,'eta':etalist,'Memory':torch.cuda.max_memory_allocated()})
