import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from data_loader import load_random_batch_pytorch, sample_random_background_pytorch

def compute_loss_sstd(nnet, xtrain, nbatch, nreal):
    loss_norm_data = np.zeros((nreal))
    loss_orth_data = np.zeros((nreal))
    loss_grad_data = np.zeros((nreal))
    for idx in range(nreal):
        xdata = Variable(load_random_batch_pytorch(xtrain, nbatch))
        loss_norm, loss_orth, loss_grad = nnet(xdata)
        loss_norm_data[idx] = loss_norm.cpu().detach().numpy()
        loss_orth_data[idx] = loss_orth.cpu().detach().numpy()
        loss_grad_data[idx] = loss_grad.cpu().detach().numpy()
    sstd_norm = np.sqrt(np.mean((loss_norm_data - np.mean(loss_norm_data)) * (loss_norm_data - np.mean(loss_norm_data))))
    sstd_orth = np.sqrt(np.mean((loss_orth_data - np.mean(loss_orth_data)) * (loss_orth_data - np.mean(loss_orth_data))))
    sstd_grad = np.sqrt(np.mean((loss_grad_data - np.mean(loss_grad_data)) * (loss_grad_data - np.mean(loss_grad_data))))
    sstd_norm = np.sqrt(nbatch) * sstd_norm
    sstd_orth = np.sqrt(nbatch) * sstd_orth
    sstd_grad = np.sqrt(nbatch) * sstd_grad
    return (sstd_norm, sstd_orth, sstd_grad)


def compute_loss_weights(loss_norm, loss_orth, loss_grad,
                         sstd_norm, sstd_orth, sstd_grad,
                         nbatch, eps=1.0e-6):
    w_orth = 1.0 / (eps + sstd_orth / np.sqrt(nbatch))
    w_norm = 1.0 / (eps + sstd_orth / np.sqrt(nbatch))
    w_grad = 1.0 / (eps + sstd_grad / np.sqrt(nbatch))
    proj_norm = torch.from_numpy((0.0 + loss_orth).cpu().detach().numpy()).double() 
    loss_vvar = torch.from_numpy((0.0 + loss_norm).cpu().detach().numpy()).double()
    w_norm = w_norm / (1.0 + np.sqrt(nbatch) / (eps + sstd_orth) * proj_norm)
    w_grad = w_grad / (1.0 + np.sqrt(nbatch) / (eps + sstd_orth) * proj_norm)
    w_grad = w_grad / (1.0 + np.sqrt(nbatch) / (eps + sstd_norm) * loss_vvar)
    w_ssum = eps + w_orth + w_norm + w_grad
    w_norm = w_norm / w_ssum
    w_orth = w_orth / w_ssum
    w_grad = w_grad / w_ssum
    return (w_norm, w_orth, w_grad)


def compute_loss_weights_simple(loss_norm, loss_orth, loss_grad,
                                nbatch, kappa=1.0e1, eps=1.0e-6):
    w_orth = 1.0
    w_norm = 1.0
    w_grad = 1.0
    proj_norm = (0.0 + loss_orth).cpu().detach().numpy()
    loss_vvar = (0.0 + loss_norm).cpu().detach().numpy()
    # proj_norm = loss_orth
    # loss_vvar = loss_norm
    w_norm = w_norm / (1.0 + np.sqrt(nbatch) * kappa * proj_norm)
    w_grad = w_grad / (1.0 + np.sqrt(nbatch) * kappa * proj_norm)
    w_grad = w_grad / (1.0 + np.sqrt(nbatch) * kappa * loss_vvar)
    w_summ = w_orth + w_norm + w_grad
    w_orth = w_orth / w_summ
    w_norm = w_norm / w_summ
    w_grad = w_grad / w_summ
    return (w_norm, w_orth, w_grad)



def compute_number_of_parameters(nnet):
    n_layers = len(nnet.fc)
    n_params = 0
    for idx in range(0, n_layers, 2):
        n_params = n_params + nnet.fc[idx].weight.size()[0] * nnet.fc[idx].weight.size()[1]
        n_params = n_params + nnet.fc[idx].weight.size()[1]
    return n_params


def flat_the_gradient(nnet):
    n_params = compute_number_of_parameters(nnet)
    n_layers = len(nnet.fc)
    grad_flat = np.zeros((n_params))
    idx_vec = 0
    for idx in range(0, n_layers, 2):
        w_grad = (nnet.fc[idx].weight.grad).cpu().detach().numpy().flatten()
        b_grad = (nnet.fc[idx].bias.grad).cpu().detach().numpy().flatten()
        grad_flat[idx_vec : (idx_vec + w_grad.size)] = w_grad.copy()
        idx_vec = idx_vec + w_grad.size
        grad_flat[idx_vec : (idx_vec + b_grad.size)] = b_grad.copy()
        idx_vec = idx_vec + b_grad.size
    return grad_flat


def compute_weights_grad_value(lv_orth, lg_orth,
                               lv_norm, lg_norm,
                               lv_grad, lg_grad,
                               kappa, eps = 1.0e-6):

    w_orth = 1.0
    w_norm = 1.0 / (1.0 + kappa * lv_orth)
    w_grad = 1.0 / (1.0 + kappa * lv_orth) / (1.0 + kappa * lv_norm)
    
    f_orth = 1.0 / np.sqrt(1.0 + np.dot(lg_orth, lg_orth))
    f_norm = 1.0 / np.sqrt(1.0 + np.dot(lg_norm, lg_norm))
    f_grad = 1.0 / np.sqrt(1.0 + np.dot(lg_grad, lg_grad))

    hmat = np.zeros((3, 3))
    hmat[0, 0] = f_orth * f_orth * np.dot(lg_orth, lg_orth)
    hmat[0, 1] = f_orth * f_norm * np.dot(lg_orth, lg_norm)
    hmat[0, 2] = f_orth * f_grad * np.dot(lg_orth, lg_grad)

    hmat[1, 0] = f_norm * f_orth * np.dot(lg_norm, lg_orth)
    hmat[1, 1] = f_norm * f_norm * np.dot(lg_norm, lg_norm)
    hmat[1, 2] = f_norm * f_grad * np.dot(lg_norm, lg_grad)

    hmat[2, 0] = f_grad * f_orth * np.dot(lg_grad, lg_orth)
    hmat[2, 1] = f_grad * f_norm * np.dot(lg_grad, lg_norm)
    hmat[2, 2] = f_grad * f_grad * np.dot(lg_grad, lg_grad)

    k_orth = w_orth - hmat[0, 1] / (hmat[0, 0] + eps) * w_norm
    factor = (hmat[0, 2] * hmat[1, 1] - hmat[0, 1] * hmat[1, 2]) / (hmat[0, 0] * hmat[1, 1] - hmat[0, 1] * hmat[0, 1] + eps)
    k_orth = k_orth + factor * w_grad
    factor = (hmat[0, 0] * hmat[1, 2] - hmat[0, 1] * hmat[0, 2]) / (hmat[0, 0] * hmat[1, 1] - hmat[0, 1] * hmat[0, 1] + eps)
    k_norm = w_norm + factor * w_grad
    factor = 0.0
    k_grad = w_grad + factor * w_grad

    k_orth = f_orth * k_orth
    k_norm = f_norm * k_norm
    k_grad = f_grad * k_grad
    return (k_orth, k_norm, k_grad)

def compute_eigen_functions(nnet, nepoch, nbatch,
                            xtrain, svec, smat,
                            lr=1.0e-4, weight_decay=1.0e-10, max_norm=1.0e-3,
                            nsample=10, nreal=100, nfreq=10,
                            variance_factor=1.0e2,
                            projection_norm_factor=1.0e2,
                            proj_deriv_norm_factor=1.0e0,
                            drct_deriv_norm_factor=1.0e0):
    nnet.fc.requires_grad_(True)
    optimizer = optim.Adam(nnet.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = optim.Adagrad(nnet.parameters(), lr=lr, weight_decay=weight_decay)
    loss_data = np.zeros((nepoch + 1))
    loss_orth_data = np.zeros((nepoch + 1))
    loss_norm_data = np.zeros((nepoch + 1))
    loss_grad_data = np.zeros((nepoch + 1))
    nweight = 0
    ntrain = xtrain.shape[0]
    n_params = compute_number_of_parameters(nnet)
    for kepoch in range(nepoch):
        nweight = min((nweight + nbatch), ntrain)
        w_bg = 1.0 / (1.0 + np.sqrt(nweight))
        xdata = Variable(load_random_batch_pytorch(xtrain, nbatch))
        zdata = Variable(sample_random_background_pytorch(nbatch, svec, smat))
        if kepoch == np.int64(0.02 * nepoch):
            print('switch optimizer')
            print('switch optimizer')
            print('switch optimizer')
            optimizer = optim.Adagrad(nnet.parameters(), lr=lr, weight_decay=weight_decay)
            print('switch optimizer')
            print('switch optimizer')
            print('switch optimizer')
        optimizer.zero_grad()
        loss_norm, loss_orth, loss_grad = nnet(xdata, zdata, w_bg)
        loss_orth_data[kepoch] = loss_orth.item()
        loss_norm_data[kepoch] = loss_norm.item()
        loss_grad_data[kepoch] = loss_grad.item()

        loss_orth.backward(retain_graph=True)
        lg_orth = flat_the_gradient(nnet)
        loss_norm.backward(retain_graph=True)
        lg_norm = flat_the_gradient(nnet)
        loss_grad.backward(retain_graph=True)
        lg_grad = flat_the_gradient(nnet)

        w_orth, w_norm, w_grad = compute_weights_grad_value(loss_orth, lg_orth,
                                                            loss_norm, lg_norm,
                                                            loss_grad, lg_grad,
                                                            1.0 * np.sqrt(nbatch),
                                                            eps=1.0e-6)
        

        loss = w_orth * loss_orth + w_norm * loss_norm + w_grad * loss_grad
        loss_data[kepoch] = loss.item()
        loss.backward()


        print('kepoch = ' + str(kepoch) + ' loss_orth.item() = ' + str(loss_orth.item()))
        print('kepoch = ' + str(kepoch) + ' loss_norm.item() = ' + str(loss_norm.item()))
        print('kepoch = ' + str(kepoch) + ' loss_grad.item() = ' + str(loss_grad.item()))
        print('kepoch = ' + str(kepoch) + ' w_orth = ' + str(w_orth))
        print('kepoch = ' + str(kepoch) + ' w_norm = ' + str(w_norm))
        print('kepoch = ' + str(kepoch) + ' w_grad = ' + str(w_grad))
        optimizer.step()

    xdata = Variable(load_random_batch_pytorch(xtrain, nbatch))
    zdata = Variable(sample_random_background_pytorch(nbatch, svec, smat))
    loss_norm, loss_orth, loss_grad = nnet(xdata, zdata, w_bg)
    loss_orth_data[-1] = loss_orth.item()
    loss_norm_data[-1] = loss_norm.item()
    loss_grad_data[-1] = loss_grad.item()

    loss_orth.backward(retain_graph=True)
    lg_orth = flat_the_gradient(nnet)
    loss_norm.backward(retain_graph=True)
    lg_norm = flat_the_gradient(nnet)
    loss_grad.backward(retain_graph=True)
    lg_grad = flat_the_gradient(nnet)

    w_orth, w_norm, w_grad = compute_weights_grad_value(loss_orth, lg_orth,
                                                        loss_norm, lg_norm,
                                                        loss_grad, lg_grad,
                                                        1.0 * np.sqrt(nbatch),
                                                        eps=1.0e-6)

    loss = w_orth * loss_orth + w_norm * loss_norm + w_grad * loss_grad
    loss_data[-1] = loss.item()
    return (loss_data, loss_orth_data, loss_norm_data, loss_grad_data)
