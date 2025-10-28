import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Model():
    def __init__(self, nnet, game, args):
        self.nnet = nnet
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        self.mse_loss = nn.MSELoss()

        if self.args['cuda']:
            self.nnet.cuda()
            self.device = next(self.nnet.parameters()).device  # <- gets device of model
        else:
            self.device = torch.device('cpu')

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=self.args['lr'])

        for epoch in range(self.args['epochs']):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / self.args['batch_size'])

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=self.args['batch_size'])
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.tensor(np.array(boards, dtype=np.float32), dtype=torch.float32, device=self.device)
                target_pis = torch.tensor(pis, dtype=torch.float32, device=self.device)
                target_vs = torch.tensor(vs, dtype=torch.float32, device=self.device)
     
                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def predict(self, board):
        board = torch.tensor(board, dtype=torch.float32, device=self.device).view(1, self.board_x, self.board_y)   # add batch dim since  predict one board
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)
        # squeeze batch dimension
        return torch.exp(pi).squeeze(0).cpu().numpy(), v.squeeze(0).cpu().numpy()

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.shape[0]

    def loss_v(self, targets, outputs):
        return self.mse_loss(outputs.view(-1), targets)


    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):

        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if self.args['cuda'] else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])