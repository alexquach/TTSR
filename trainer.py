from utils import calc_psnr_and_ssim
from model import Vgg19

import os
import numpy as np
from imageio import imread, imsave
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)


def naive_averaging(model, lr, lr_sr, hr, ref, ref_sr):
    sr_list = []
    S_list = []
    T_lv3_list = []
    T_lv2_list = []
    T_lv1_list = []

    # TODO: Make sure gradients propagate, aka not doing non-differentiable things

    # for each frame
    for i in range(5):
        # get TTSR output for each frame
        a, b, c, d, e = model(
            lr=lr[:, [i], :, :].repeat(1, 3, 1, 1),
            lrsr=lr_sr[:, [i], :, :].repeat(1, 3, 1, 1),
            ref=ref.repeat(1, 3, 1, 1),
            refsr=ref_sr.repeat(1, 3, 1, 1)
        )

        sr_list.append(a)
        S_list.append(b)
        T_lv3_list.append(c)
        T_lv2_list.append(d)
        T_lv1_list.append(e)

    # Stack: [frame, batch_size, channels, height, width]
    # Mean: [batch_size, channels, height, width]
    sr = torch.mean(torch.stack(sr_list, dim=0), dim=0)
    S = torch.mean(torch.stack(S_list, dim=0), dim=0)
    T_lv3 = torch.mean(torch.stack(T_lv3_list, dim=0), dim=0)
    T_lv2 = torch.mean(torch.stack(T_lv2_list, dim=0), dim=0)
    T_lv1 = torch.mean(torch.stack(T_lv1_list, dim=0), dim=0)

    # sr = torch.mean(torch.cat(sr_list, dim=0), dim=0)
    # S = torch.mean(torch.cat(S_list, dim=0), dim=0)
    # T_lv3 = torch.mean(torch.cat(T_lv3_list, dim=0), dim=0)
    # T_lv2 = torch.mean(torch.cat(T_lv2_list, dim=0), dim=0)
    # T_lv1 = torch.mean(torch.cat(T_lv1_list, dim=0), dim=0)

    # enforce backprop on the variables
    sr = Variable(sr.data, requires_grad=True)
    S = Variable(S.data, requires_grad=True)
    T_lv3 = Variable(T_lv3.data, requires_grad=True)
    T_lv2 = Variable(T_lv2.data, requires_grad=True)
    T_lv1 = Variable(T_lv1.data, requires_grad=True)

    return sr, S, T_lv3, T_lv2, T_lv1


def flownet_naive_averaging(model, lr, lr_sr, hr, ref, ref_sr):
    sr_list = torch.tensor([], requires_grad=True).cuda()
    S_list = torch.tensor([], requires_grad=True).cuda()
    T_lv3_list = torch.tensor([], requires_grad=True).cuda()
    T_lv2_list = torch.tensor([], requires_grad=True).cuda()
    T_lv1_list = torch.tensor([], requires_grad=True).cuda()

    # TODO: Make sure gradients propagate, aka not doing non-differentiable things

    # for each frame
    for i in range(5):
        # get TTSR output for each frame
        a, b, c, d, e = model(
            lr=lr[:, i, :, :, :],
            lrsr=lr_sr[:, i, :, :, :],
            ref=ref[:, :, :, :],
            refsr=ref_sr[:, :, :, :])
        
        sr_list = torch.cat([sr_list, a.unsqueeze(0)])
        S_list = torch.cat([S_list, b.unsqueeze(0)])
        T_lv3_list = torch.cat([T_lv3_list, c.unsqueeze(0)])
        T_lv2_list = torch.cat([T_lv2_list, d.unsqueeze(0)])
        T_lv1_list = torch.cat([T_lv1_list, e.unsqueeze(0)])

    # Stack: [frame, batch_size, channels, height, width]
    # Mean: [batch_size, channels, height, width]
    sr = torch.mean(sr_list, dim=0)
    S = torch.mean(S_list, dim=0)
    T_lv3 = torch.mean(T_lv3_list, dim=0)
    T_lv2 = torch.mean(T_lv2_list, dim=0)
    T_lv1 = torch.mean(T_lv1_list, dim=0)

    # # enforce backprop on the variables
    # sr = Variable(sr.data, requires_grad=True)
    # S = Variable(S.data, requires_grad=True)
    # T_lv3 = Variable(T_lv3.data, requires_grad=True)
    # T_lv2 = Variable(T_lv2.data, requires_grad=True)
    # T_lv1 = Variable(T_lv1.data, requires_grad=True)

    return sr, S, T_lv3, T_lv2, T_lv1


def flownet_conv3d_1x1(model, lr, lr_sr, hr, ref, ref_sr):
    sr_list = torch.tensor([], requires_grad=True).cuda()
    S_list = torch.tensor([], requires_grad=True).cuda()
    T_lv3_list = torch.tensor([], requires_grad=True).cuda()
    T_lv2_list = torch.tensor([], requires_grad=True).cuda()
    T_lv1_list = torch.tensor([], requires_grad=True).cuda()

    # for each frame
    for i in range(5):
        # get TTSR output for each frame
        a, b, c, d, e = model(
            lr=lr[:, i, :, :, :],
            lrsr=lr_sr[:, i, :, :, :],
            ref=ref[:, :, :, :],
            refsr=ref_sr[:, :, :, :])
        
        sr_list = torch.cat([sr_list, a.unsqueeze(0)])
        S_list = torch.cat([S_list, b.unsqueeze(0)])
        T_lv3_list = torch.cat([T_lv3_list, c.unsqueeze(0)])
        T_lv2_list = torch.cat([T_lv2_list, d.unsqueeze(0)])
        T_lv1_list = torch.cat([T_lv1_list, e.unsqueeze(0)])

    #[-1, 1] or [-0.5, 0.5]

    # Stack: [frames, batch_size, channels, height, width]
    # i.e.: [5, 9, 3, 160, 160]
    # resu: [9, 5, 3, 160, 160]
    # conv: [N, C, D, H, W]
    sr_list = sr_list.permute(1, 0, 2, 3, 4)
    S_list = S_list.permute(1, 0, 2, 3, 4)
    T_lv3_list = T_lv3_list.permute(1, 0, 2, 3, 4)
    T_lv2_list = T_lv2_list.permute(1, 0, 2, 3, 4)
    T_lv1_list = T_lv1_list.permute(1, 0, 2, 3, 4)

    # # from: [9, 5, 3, 160, 160]
    # conv1 = nn.Conv3d(5, 5, 1, stride=1).cuda()
    # sr_list = torch.tanh(conv1(sr_list))
    # S_list = torch.tanh(conv1(S_list))
    # T_lv3_list = torch.tanh(conv1(T_lv3_list))
    # T_lv2_list = torch.tanh(conv1(T_lv2_list))
    # T_lv1_list = torch.tanh(conv1(T_lv1_list))
    #   to: [9, 5, 3, 160, 160]
    conv2 = nn.Conv3d(5, 1, 1, stride=1).cuda()
    sr_list = torch.tanh(conv2(sr_list))
    S_list = torch.tanh(conv2(S_list))
    T_lv3_list = torch.tanh(conv2(T_lv3_list))
    T_lv2_list = torch.tanh(conv2(T_lv2_list))
    T_lv1_list = torch.tanh(conv2(T_lv1_list))
    #   to: [9, 1, 3, 160, 160]
    sr = sr_list.squeeze(1)
    S = S_list.squeeze(1)
    T_lv3 = T_lv3_list.squeeze(1)
    T_lv2 = T_lv2_list.squeeze(1)
    T_lv1 = T_lv1_list.squeeze(1)
    #   to: [9, 3, 160, 160]

    return sr, S, T_lv3, T_lv2, T_lv1

def conv1x1_fusion(model, lr, lr_sr, hr, ref, ref_sr):
    sr_list = []
    S_list = []
    T_lv3_list = []
    T_lv2_list = []
    T_lv1_list = []

    # for each frame
    for i in range(5):
        # get TTSR output for each frame
        a, b, c, d, e = model(
            lr=lr[:, [i], :, :].repeat(1, 3, 1, 1),
            lrsr=lr_sr[:, [i], :, :].repeat(1, 3, 1, 1),
            ref=ref.repeat(1, 3, 1, 1),
            refsr=ref_sr.repeat(1, 3, 1, 1)
        )

        sr_list.append(a)
        S_list.append(b)
        T_lv3_list.append(c)
        T_lv2_list.append(d)
        T_lv1_list.append(e)



    return sr, S, T_lv3, T_lv2, T_lv1


class Trainer():
    def __init__(self, args, logger, dataloader, model, loss_all):
        self.args = args
        self.logger = logger
        self.dataloader = dataloader
        self.model = model
        self.loss_all = loss_all
        self.device = torch.device('cpu') if args.cpu else torch.device('cuda')
        self.vgg19 = Vgg19.Vgg19(requires_grad=False).to(self.device)
        if ((not self.args.cpu) and (self.args.num_gpu > 1)):
            self.vgg19 = nn.DataParallel(
                self.vgg19, list(range(self.args.num_gpu)))

        self.params = [
            {"params": filter(lambda p: p.requires_grad, self.model.MainNet.parameters() if
                              args.num_gpu == 1 else self.model.module.MainNet.parameters()),
             "lr": args.lr_rate
             },
            {"params": filter(lambda p: p.requires_grad, self.model.LTE.parameters() if
                              args.num_gpu == 1 else self.model.module.LTE.parameters()),
             "lr": args.lr_rate_lte
             }
        ]
        self.optimizer = optim.Adam(self.params, betas=(
            args.beta1, args.beta2), eps=args.eps)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.args.decay, gamma=self.args.gamma)
        self.max_psnr = 0.
        self.max_psnr_epoch = 0
        self.max_ssim = 0.
        self.max_ssim_epoch = 0

    def load(self, model_path=None):
        if (model_path):
            self.logger.info('load_model_path: ' + model_path)
            #model_state_dict_save = {k.replace('module.',''):v for k,v in torch.load(model_path).items()}
            model_state_dict_save = {k: v for k, v in torch.load(
                model_path, map_location=self.device).items()}
            model_state_dict = self.model.state_dict()
            model_state_dict.update(model_state_dict_save)
            self.model.load_state_dict(model_state_dict)

    def prepare(self, sample_batched):
        for key in sample_batched.keys():
            sample_batched[key] = sample_batched[key].to(self.device)
        return sample_batched

    def train(self, current_epoch=0, is_init=False):
        self.model.train()
        if (not is_init):
            self.scheduler.step()
        self.logger.info('Current epoch learning rate: %e' %
                         (self.optimizer.param_groups[0]['lr']))

        for i_batch, sample_batched in enumerate(self.dataloader['train']):
            self.optimizer.zero_grad()

            sample_batched = self.prepare(sample_batched)
            lr = sample_batched['LR']
            lr_sr = sample_batched['LR_sr']
            hr = sample_batched['HR']
            ref = sample_batched['Ref']
            ref_sr = sample_batched['Ref_sr']

            sr, S, T_lv3, T_lv2, T_lv1 = self.model(
                lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr)
            # hr = hr.repeat(1, 3, 1, 1)
            # TODO: make better fusion module
            # sr, S, T_lv3, T_lv2, T_lv1 = naive_averaging(
            # self.model, lr, lr_sr, hr, ref, ref_sr)

            # calc loss
            is_print = ((i_batch + 1) %
                        self.args.print_every == 0)  # flag of print

            rec_loss = self.args.rec_w * self.loss_all['rec_loss'](sr, hr)
            loss = rec_loss
            if (is_print):
                self.logger.info(('init ' if is_init else '') + 'epoch: ' + str(current_epoch) +
                                 '\t batch: ' + str(i_batch+1))
                self.logger.info('rec_loss: %.10f' % (rec_loss.item()))

            if (not is_init):
                if ('per_loss' in self.loss_all):
                    sr_relu5_1 = self.vgg19((sr + 1.) / 2.)
                    with torch.no_grad():
                        hr_relu5_1 = self.vgg19((hr.detach() + 1.) / 2.)
                    per_loss = self.args.per_w * \
                        self.loss_all['per_loss'](sr_relu5_1, hr_relu5_1)
                    loss += per_loss
                    if (is_print):
                        self.logger.info('per_loss: %.10f' % (per_loss.item()))
                if ('tpl_loss' in self.loss_all):
                    sr_lv1, sr_lv2, sr_lv3 = self.model(sr=sr)
                    tpl_loss = self.args.tpl_w * self.loss_all['tpl_loss'](sr_lv3, sr_lv2, sr_lv1,
                                                                           S, T_lv3, T_lv2, T_lv1)
                    loss += tpl_loss
                    if (is_print):
                        self.logger.info('tpl_loss: %.10f' % (tpl_loss.item()))
                if ('adv_loss' in self.loss_all):
                    adv_loss = self.args.adv_w * \
                        self.loss_all['adv_loss'](sr, hr)
                    loss += adv_loss
                    if (is_print):
                        self.logger.info('adv_loss: %.10f' % (adv_loss.item()))

            loss.backward()
            self.optimizer.step()

        if ((not is_init) and current_epoch % self.args.save_every == 0):
            self.logger.info('saving the model...')
            tmp = self.model.state_dict()
            model_state_dict = {key.replace('module.', ''): tmp[key] for key in tmp if
                                (('SearchNet' not in key) and ('_copy' not in key))}
            model_name = self.args.save_dir.strip(
                '/')+'/model/model_'+str(current_epoch).zfill(5)+'.pt'
            torch.save(model_state_dict, model_name)

    def evaluate(self, current_epoch=0):
        self.logger.info('Epoch ' + str(current_epoch) +
                         ' evaluation process...')

        if (self.args.dataset == 'HMDB_FRAMES'):
            self.model.eval()
            with torch.no_grad():
                psnr, ssim, cnt = 0., 0., 0
                for i_batch, sample_batched in enumerate(self.dataloader['test']['1']):
                    cnt += 1
                    sample_batched = self.prepare(sample_batched)
                    lr = sample_batched['LR']
                    lr_sr = sample_batched['LR_sr']
                    hr = sample_batched['HR']
                    ref = sample_batched['Ref']
                    ref_sr = sample_batched['Ref_sr']
                    hr = hr.repeat(1, 3, 1, 1)

                    # sr, _, _, _, _ = naive_averaging(
                    # self.model, lr, lr_sr, hr, ref, ref_sr) TODO uncomment for modified fusion model
                    sr, _, _, _, _ = self.model(
                        lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr)

                    if (self.args.eval_save_results):
                        sr_save = (sr+1.) * 127.5
                        sr_save = np.transpose(sr_save.squeeze().round(
                        ).cpu().numpy(), (1, 2, 0)).astype(np.uint8)
                        imsave(os.path.join(self.args.save_dir, 'save_results', str(
                            i_batch).zfill(5)+'.png'), sr_save)

                    # calculate psnr and ssim
                    # sr = sr.squeeze(0)
                    # hr = hr.squeeze(0) TODO uncomment for modified fusion model
                    _psnr, _ssim = calc_psnr_and_ssim(sr.detach(), hr.detach())

                    psnr += _psnr
                    ssim += _ssim

                psnr_ave = psnr / cnt
                ssim_ave = ssim / cnt
                self.logger.info(
                    'Ref  PSNR (now): %.3f \t SSIM (now): %.4f' % (psnr_ave, ssim_ave))
                if (psnr_ave > self.max_psnr):
                    self.max_psnr = psnr_ave
                    self.max_psnr_epoch = current_epoch
                if (ssim_ave > self.max_ssim):
                    self.max_ssim = ssim_ave
                    self.max_ssim_epoch = current_epoch
                self.logger.info('Ref  PSNR (max): %.3f (%d) \t SSIM (max): %.4f (%d)'
                                 % (self.max_psnr, self.max_psnr_epoch, self.max_ssim, self.max_ssim_epoch))

        self.logger.info('Evaluation over.')

    def test(self):
        self.logger.info('Test process...')
        self.logger.info('lr path:     %s' % (self.args.lr_path))
        self.logger.info('ref path:    %s' % (self.args.ref_path))

        ### LR and LR_sr
        LR = imread(self.args.lr_path)
        h1, w1 = LR.shape[:2]
        LR_sr = np.array(Image.fromarray(
            LR).resize((w1*4, h1*4), Image.BICUBIC))

        ### Ref and Ref_sr
        Ref = imread(self.args.ref_path)
        h2, w2 = Ref.shape[:2]
        h2, w2 = h2//4*4, w2//4*4
        Ref = Ref[:h2, :w2, :]
        Ref_sr = np.array(Image.fromarray(Ref).resize(
            (w2//4, h2//4), Image.BICUBIC))
        Ref_sr = np.array(Image.fromarray(
            Ref_sr).resize((w2, h2), Image.BICUBIC))

        # change type
        LR = LR.astype(np.float32)
        LR_sr = LR_sr.astype(np.float32)
        Ref = Ref.astype(np.float32)
        Ref_sr = Ref_sr.astype(np.float32)

        # rgb range to [-1, 1]
        LR = LR / 127.5 - 1.
        LR_sr = LR_sr / 127.5 - 1.
        Ref = Ref / 127.5 - 1.
        Ref_sr = Ref_sr / 127.5 - 1.

        # to tensor
        LR_t = torch.from_numpy(LR.transpose((2, 0, 1))).unsqueeze(
            0).float().to(self.device)
        LR_sr_t = torch.from_numpy(LR_sr.transpose(
            (2, 0, 1))).unsqueeze(0).float().to(self.device)
        Ref_t = torch.from_numpy(Ref.transpose((2, 0, 1))).unsqueeze(
            0).float().to(self.device)
        Ref_sr_t = torch.from_numpy(Ref_sr.transpose(
            (2, 0, 1))).unsqueeze(0).float().to(self.device)

        self.model.eval()
        with torch.no_grad():
            sr, _, _, _, _ = self.model(
                lr=LR_t, lrsr=LR_sr_t, ref=Ref_t, refsr=Ref_sr_t)
            sr_save = (sr+1.) * 127.5
            sr_save = np.transpose(sr_save.squeeze().round(
            ).cpu().numpy(), (1, 2, 0)).astype(np.uint8)
            save_path = os.path.join(
                self.args.save_dir, 'save_results', os.path.basename(self.args.lr_path))
            imsave(save_path, sr_save)
            self.logger.info('output path: %s' % (save_path))

        self.logger.info('Test over.')

    def train_flownet(self, args, current_epoch=0, is_init=False):
        self.model.train()
        if (not is_init):
            self.scheduler.step()
        self.logger.info('Current epoch learning rate: %e' %
                         (self.optimizer.param_groups[0]['lr']))

        for i_batch, sample_batched in enumerate(self.dataloader['train']):
            self.optimizer.zero_grad()

            sample_batched = self.prepare(sample_batched)
            lr = sample_batched['LR'].float() / 255.0           # [9, 5, 3, 40, 40]  
            lr_sr = sample_batched['LR_sr'].float() / 255.0     # [9, 5, 3, 160, 160]
            hr = sample_batched['HR'].float() / 255.0           # [9, 3, 160, 160]
            ref = sample_batched['Ref'].float() / 255.0         # [9, 3, 160, 160]
            ref_sr = sample_batched['Ref_sr'].float() / 255.0   # [9, 3, 160, 160]

            # #input to self.model:
            # lr       [9, 3, 40, 40]
            # lr_sr    [9, 3, 160, 160]
            # ref      [9, 3, 160, 160]
            # ref_sr   [9, 3, 160, 160]

            # TODO: make better fusion module
            if args.train_style == "normal":
                sr, S, T_lv3, T_lv2, T_lv1 = self.model(
                    lr=lr[:, 2, :, :, :],
                    lrsr=lr_sr[:, 2, :, :, :],
                    ref=ref[:, :, :, :],
                    refsr=ref_sr[:, :, :, :])
            elif args.train_style == "average":
                sr, S, T_lv3, T_lv2, T_lv1 = flownet_naive_averaging(
                    self.model, lr, lr_sr, hr, ref, ref_sr)
            else:
                sr, S, T_lv3, T_lv2, T_lv1 = flownet_conv3d_1x1(
                    self.model, lr, lr_sr, hr, ref, ref_sr)

            # calc loss
            is_print = ((i_batch + 1) %
                        self.args.print_every == 0)  # flag of print

            rec_loss = self.args.rec_w * self.loss_all['rec_loss'](sr, hr)
            loss = rec_loss
            if (is_print):
                self.logger.info(('init ' if is_init else '') + 'epoch: ' + str(current_epoch) +
                                 '\t batch: ' + str(i_batch+1))
                self.logger.info('rec_loss: %.10f' % (rec_loss.item()))

            if (not is_init):
                if ('per_loss' in self.loss_all):
                    if args.lpips:
                        per_loss = self.args.per_w * self.loss_all['per_loss'](sr, hr) # no adjustment because lpips requires [-1, 1]
                    else:
                        sr_relu5_1 = self.vgg19((sr + 1.) / 2.)
                        with torch.no_grad():
                            hr_relu5_1 = self.vgg19((hr.detach() + 1.) / 2.)
                        per_loss = self.args.per_w * \
                            self.loss_all['per_loss'](sr_relu5_1, hr_relu5_1)
                    loss += per_loss
                    if (is_print):
                        self.logger.info('per_loss: %.10f' % (per_loss.item()))
                if ('tpl_loss' in self.loss_all):
                    sr_lv1, sr_lv2, sr_lv3 = self.model(sr=sr)
                    tpl_loss = self.args.tpl_w * self.loss_all['tpl_loss'](sr_lv3, sr_lv2, sr_lv1,
                                                                           S, T_lv3, T_lv2, T_lv1)
                    loss += tpl_loss
                    if (is_print):
                        self.logger.info('tpl_loss: %.10f' % (tpl_loss.item()))
                if ('adv_loss' in self.loss_all):
                    adv_loss = self.args.adv_w * \
                        self.loss_all['adv_loss'](sr, hr)
                    loss += adv_loss
                    if (is_print):
                        self.logger.info('adv_loss: %.10f' % (adv_loss.item()))

            loss.backward()
            self.optimizer.step()

        if ((not is_init) and current_epoch % self.args.save_every == 0):
            self.logger.info('saving the model...')
            tmp = self.model.state_dict()
            model_state_dict = {key.replace('module.', ''): tmp[key] for key in tmp if
                                (('SearchNet' not in key) and ('_copy' not in key))}
            model_name = self.args.save_dir.strip(
                '/')+'/model/model_'+str(current_epoch).zfill(5)+'.pt'
            torch.save(model_state_dict, model_name)

    def evaluate_flownet(self, args, current_epoch=0):
        self.logger.info('Epoch ' + str(current_epoch) +
                         ' evaluation process...')

        if (self.args.dataset == 'HMDB_FLOWNET'):
            self.model.eval()
            with torch.no_grad():
                psnr, ssim, cnt = 0., 0., 0
                for i_batch, sample_batched in enumerate(self.dataloader['test']['1']):
                    cnt += 1
                    sample_batched = self.prepare(sample_batched)
                    lr = sample_batched['LR'].float() / 255.0           # [5, 3, 40, 40]  
                    lr_sr = sample_batched['LR_sr'].float() / 255.0     # [5, 3, 160, 160]
                    hr = sample_batched['HR'].float() / 255.0           # [1, 3, 160, 160]
                    ref = sample_batched['Ref'].float() / 255.0         # [1, 3, 160, 160]
                    ref_sr = sample_batched['Ref_sr'].float() / 255.0   # [1, 3, 160, 160]

                    if args.train_style == "normal":
                        sr, S, T_lv3, T_lv2, T_lv1 = self.model(
                            lr=lr[:, 2, :, :, :],
                            lrsr=lr_sr[:, 2, :, :, :],
                            ref=ref[:, :, :, :],
                            refsr=ref_sr[:, :, :, :])
                    elif args.train_style == "average":
                        sr, S, T_lv3, T_lv2, T_lv1 = flownet_naive_averaging(
                            self.model, lr, lr_sr, hr, ref, ref_sr)
                    else:
                        sr, S, T_lv3, T_lv2, T_lv1 = flownet_conv3d_1x1(
                            self.model, lr, lr_sr, hr, ref, ref_sr)

                    if (self.args.eval_save_results):
                        sr_save = (sr+1.) * 127.5
                        sr_save = np.transpose(sr_save.squeeze().round(
                        ).cpu().numpy(), (1, 2, 0)).astype(np.uint8)
                        imsave(os.path.join(self.args.save_dir, 'save_results', str(
                            i_batch).zfill(5)+'.png'), sr_save)

                    # calculate psnr and ssim
                    # sr = sr.squeeze(0)
                    # hr = hr.squeeze(0) TODO uncomment for modified fusion model
                    _psnr, _ssim = calc_psnr_and_ssim(sr.detach(), hr.detach())

                    psnr += _psnr
                    ssim += _ssim

                psnr_ave = psnr / cnt
                ssim_ave = ssim / cnt
                self.logger.info(
                    'Ref  PSNR (now): %.3f \t SSIM (now): %.4f' % (psnr_ave, ssim_ave))
                if (psnr_ave > self.max_psnr):
                    self.max_psnr = psnr_ave
                    self.max_psnr_epoch = current_epoch
                if (ssim_ave > self.max_ssim):
                    self.max_ssim = ssim_ave
                    self.max_ssim_epoch = current_epoch
                self.logger.info('Ref  PSNR (max): %.3f (%d) \t SSIM (max): %.4f (%d)'
                                 % (self.max_psnr, self.max_psnr_epoch, self.max_ssim, self.max_ssim_epoch))

        self.logger.info('Evaluation over.')

    def test_flownet(self):
        self.logger.info('Test process...')
        self.logger.info('lr path:     %s' % (self.args.lr_path))
        self.logger.info('ref path:    %s' % (self.args.ref_path))

        ### LR and LR_sr
        LR = imread(self.args.lr_path)
        h1, w1 = LR.shape[:2]
        LR_sr = np.array(Image.fromarray(
            LR).resize((w1*4, h1*4), Image.BICUBIC))

        ### Ref and Ref_sr
        Ref = imread(self.args.ref_path)
        h2, w2 = Ref.shape[:2]
        h2, w2 = h2//4*4, w2//4*4
        Ref = Ref[:h2, :w2, :]
        Ref_sr = np.array(Image.fromarray(Ref).resize(
            (w2//4, h2//4), Image.BICUBIC))
        Ref_sr = np.array(Image.fromarray(
            Ref_sr).resize((w2, h2), Image.BICUBIC))

        # change type
        LR = LR.astype(np.float32)
        LR_sr = LR_sr.astype(np.float32)
        Ref = Ref.astype(np.float32)
        Ref_sr = Ref_sr.astype(np.float32)

        # rgb range to [-1, 1]
        LR = LR / 127.5 - 1.
        LR_sr = LR_sr / 127.5 - 1.
        Ref = Ref / 127.5 - 1.
        Ref_sr = Ref_sr / 127.5 - 1.

        # to tensor
        LR_t = torch.from_numpy(LR.transpose((2, 0, 1))).unsqueeze(
            0).float().to(self.device)
        LR_sr_t = torch.from_numpy(LR_sr.transpose(
            (2, 0, 1))).unsqueeze(0).float().to(self.device)
        Ref_t = torch.from_numpy(Ref.transpose((2, 0, 1))).unsqueeze(
            0).float().to(self.device)
        Ref_sr_t = torch.from_numpy(Ref_sr.transpose(
            (2, 0, 1))).unsqueeze(0).float().to(self.device)

        self.model.eval()
        with torch.no_grad():
            sr, _, _, _, _ = self.model(
                lr=LR_t, lrsr=LR_sr_t, ref=Ref_t, refsr=Ref_sr_t)
            sr_save = (sr+1.) * 127.5
            sr_save = np.transpose(sr_save.squeeze().round(
            ).cpu().numpy(), (1, 2, 0)).astype(np.uint8)
            save_path = os.path.join(
                self.args.save_dir, 'save_results', os.path.basename(self.args.lr_path))
            imsave(save_path, sr_save)
            self.logger.info('output path: %s' % (save_path))

        self.logger.info('Test over.')