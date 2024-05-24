from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, LeRet
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.loss=args.loss
    def _build_model(self):
        model_dict = {
            'DLinear': DLinear,
            'LeRet':LeRet,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        self.train_data, self.train_loader = self._get_data(flag='train')
        self.vali_data, self.vali_loader = self._get_data(flag='val')
        self.test_data, self.test_loader = self._get_data(flag='test')
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.loss == 'mae':
            print('mae loss')
            criterion = nn.L1Loss()
        else:
            print('mse loss')
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_auto_loss = []
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,y_auto) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                y_auto = y_auto.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                outputs,pred_auto =  self.model(batch_x)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)
                auto_loss = criterion(pred_auto, y_auto)

                total_loss.append(loss.item())
                total_auto_loss.append(auto_loss.item())
        total_loss = np.average(total_loss)
        total_auto_loss = np.average(total_auto_loss)
        self.model.train()
        return total_loss,total_auto_loss

    def auto_regression_pretrain(self,setting):
        train_data, train_loader = self.train_data, self.train_loader
        vali_data, vali_loader = self.vali_data, self.vali_loader
        test_data, test_loader = self.test_data, self.test_loader

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)
        save_file_name = 'autopretrain-checkpoint.pth'
        for epoch in range(self.args.pretrain_epochs):
            iter_count = 0
            train_loss = []
            train_auto_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,y_auto) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                y_auto = y_auto.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
             
                outputs,pred_auto =  self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                loss = criterion(outputs, batch_y)
                auto_loss = criterion(pred_auto, y_auto)
                back_loss =  auto_loss # 进行自回归损失预训练
                train_loss.append(loss.item())
                train_auto_loss.append(auto_loss.item())
                
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | pred_loss: {2:.7f} | auto_loss {3:.7f} | back_loss {4:.7f}".format(i + 1, epoch + 1, loss.item(),auto_loss.item(),back_loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

    
                back_loss.backward()
                model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_auto_loss = np.average(train_auto_loss)
            vali_loss,vali_auto_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss,test_auto_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Predict | Predict Train : {2:.7f} Predict Vali : {3:.7f} Predict Test : {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            print("Epoch: {0}, Steps: {1} | AutoReg | Train_auto_loss : {2:.7f} Vali_auto_loss: {3:.7f} Test_auto_loss: {4:.7f}".format(
                epoch + 1, train_steps, train_auto_loss, vali_auto_loss, test_auto_loss))
            print('Epoch: {0}, Steps: {1} | Back_loss: {2:.7f}'.format( epoch + 1, train_steps,train_auto_loss+train_loss))
            early_stopping(vali_auto_loss, self.model, path,save_file_name)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + save_file_name
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model    

    def train(self, setting):
        print(f'Auto regression train starts | Run {self.args.pretrain_epochs} Epochs')
        self.auto_regression_pretrain(setting)
        print(f'Auto regression train ends | Run {self.args.pretrain_epochs} Epochs')
        train_data, train_loader = self.train_data, self.train_loader
        vali_data, vali_loader = self.vali_data, self.vali_loader
        test_data, test_loader = self.test_data, self.test_loader

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_auto_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,y_auto) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                y_auto = y_auto.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

    
                outputs,pred_auto =  self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                auto_loss = criterion(pred_auto, y_auto)
                back_loss = loss # 采取两阶段训练方式
                #back_loss = loss # + auto_loss # 按照1:1来的，感觉可能不太行
                train_loss.append(loss.item())
                train_auto_loss.append(auto_loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | pred_loss: {2:.7f} | auto_loss {3:.7f} | back_loss {4:.7f}".format(i + 1, epoch + 1, loss.item(),auto_loss.item(),back_loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(back_loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    back_loss.backward()
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_auto_loss = np.average(train_auto_loss)
            vali_loss,vali_auto_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss,test_auto_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Predict | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            print("Epoch: {0}, Steps: {1} | AutoReg | Train_auto_loss : {2:.7f} Vali_auto_loss: {3:.7f} Test_auto_loss: {4:.7f}".format(
                epoch + 1, train_steps, train_auto_loss, vali_auto_loss, test_auto_loss))
            print('Epoch: {0}, Steps: {1} | Back_loss: {2:.7f}'.format( epoch + 1, train_steps,train_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,y_auto) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                y_auto = y_auto.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs,pred_auto =  self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                
                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
                # if 'PEMS' in setting:
                #     print(pred.shape,true.shape)
                #     pred = test_data.inverse_transform(pred)
                #     true = test_data.inverse_transform(true)
                preds.append(pred)
                trues.append(true)
                # inputx.append(batch_x.detach().cpu().numpy())
        #         if i % 20 == 0:
        #             input = batch_x.detach().cpu().numpy()
        #             gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
        #             pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
        #             visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # if self.args.test_flop:
        #     test_params_flop((batch_x.shape[1],batch_x.shape[2]))
        #     exit()
        preds = np.array(preds)
        trues = np.array(trues)
        # inputx = np.array(inputx)

        # inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        
        
        if 'PEMS' in setting:
            preds = preds.reshape(-1,preds.shape[-1])
            trues = trues.reshape(-1,trues.shape[-1])
            preds = test_data.inverse_transform(preds)
            trues = test_data.inverse_transform(trues)
            mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
            print('mae:{}, mape:{}, rmse:{}'.format(mae, mape, rmse))

        else:
            
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
            print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        # f = open("result.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        # # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    # def predict(self, setting, load=False):
    #     pred_data, pred_loader = self._get_data(flag='pred')

    #     if load:
    #         path = os.path.join(self.args.checkpoints, setting)
    #         best_model_path = path + '/' + 'checkpoint.pth'
    #         self.model.load_state_dict(torch.load(best_model_path))

    #     preds = []

    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,y_auto) in enumerate(pred_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float()
    #             batch_x_mark = batch_x_mark.float().to(self.device)
    #             batch_y_mark = batch_y_mark.float().to(self.device)

    #             # decoder input
    #             dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
    #             dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
    #             # encoder - decoder
    #             if self.args.use_amp:
    #                 with torch.cuda.amp.autocast():
    #                     if 'Linear' in self.args.model or 'TST' in self.args.model:
    #                         outputs = self.model(batch_x)
    #                     else:
    #                         if self.args.output_attention:
    #                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    #                         else:
    #                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #             else:
    #                 if 'Linear' in self.args.model or 'TST' in self.args.model:
    #                     outputs = self.model(batch_x)
    #                 else:
    #                     if self.args.output_attention:
    #                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    #                     else:
    #                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #             pred = outputs.detach().cpu().numpy()  # .squeeze()
    #             preds.append(pred)

    #     preds = np.array(preds)
    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

    #     # result save
    #     folder_path = './results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    #     np.save(folder_path + 'real_prediction.npy', preds)

    #     return
