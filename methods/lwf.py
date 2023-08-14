import numpy as np
import torch
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from models.base import BaseLearner
from utils.data_manager import partition_data, DatasetSplit, average_weights, setup_seed
import wandb, copy

# init_epoch = 200
# epochs = 250
# self.args["com_round"] = 100
# self.args["num_users"] = 5
# self.args["frac"] = 1
# self.args["local_bs"] = 128
# self.args["local_ep"] = 5
# batch_size = 128
# num_workers = 4

# init_epoch = 5
# init_lr = 0.1
# init_milestones = [60, 120, 160]
# init_lr_decay = 0.1
# init_weight_decay = 0.0005



# epochs = 5
# lrate = 0.1
# milestones = [60, 120, 180, 220]
# lrate_decay = 0.1
# batch_size = 128
# weight_decay = 2e-4
# num_workers = 8

T = 2
lamda = 3


class LwF(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)   # last layer for 10-classes classification
        print("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(   # return dataset, labels 0:9
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        # self.train_loader = DataLoader(
        #     train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        # )
        # import pdb
        # pdb.set_trace()
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        # if self.args["dataset"] != "tiny_imagenet":

        self.test_loader = DataLoader(
            test_dataset, batch_size=256, shuffle=False, num_workers=4
        )

        setup_seed(self.seed)
        self._fl_train(train_dataset, self.test_loader)

        # if len(self._multiple_gpus) > 1:
        #     self._network = nn.DataParallel(self._network, self._multiple_gpus)
        # self._train(self.train_loader, self.test_loader)    #* train the task
        # if len(self._multiple_gpus) > 1:
        #     self._network = self._network.module

    def _local_update(self, model, train_data_loader):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        for iter in range(self.args["local_ep"]):
            # import pdb
            # pdb.set_trace()
            for batch_idx, (_, images, labels) in enumerate(train_data_loader):
                images, labels = images.cuda(), labels.cuda()
                output = model(images)["logits"]
                loss = F.cross_entropy(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model.state_dict()

    def _local_finetune(self, model, train_data_loader):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        for iter in range(self.args["local_ep"]):
            for batch_idx, (_, images, labels) in enumerate(train_data_loader):
                images, labels = images.cuda(), labels.cuda()
                fake_targets = labels - self._known_classes
                output = model(images)["logits"]
                loss_clf = F.cross_entropy(
                    output[:, self._known_classes :], fake_targets  # logits[10:20] -- 0~9 class
                )
                loss_kd = _KD_loss(
                    output[:, : self._known_classes],   # logits on previous tasks
                    self._old_network(images)["logits"],
                    T,
                )
                loss = lamda * loss_kd + loss_clf # 25 * loss_kd + loss_clf # lamda * loss_kd + loss_clf
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model.state_dict()


    def _fl_train(self, train_dataset, test_loader):
        self._network.cuda()
        user_groups = partition_data(train_dataset.labels, beta=self.args["beta"], n_parties=self.args["num_users"])
        prog_bar = tqdm(range(self.args["com_round"]))
        for _, com in enumerate(prog_bar):
            local_weights = []
            m = max(int(self.args["frac"] * self.args["num_users"]), 1)
            idxs_users = np.random.choice(range(self.args["num_users"]), m, replace=False)
            for idx in idxs_users:
                local_train_loader = DataLoader(DatasetSplit(train_dataset, user_groups[idx]), 
                    batch_size=self.args["local_bs"], shuffle=True, num_workers=4)
                if self._cur_task == 0:
                    w = self._local_update(copy.deepcopy(self._network), local_train_loader)
                else:
                    w = self._local_finetune(copy.deepcopy(self._network), local_train_loader)
                local_weights.append(copy.deepcopy(w))
            # update global weights
            global_weights = average_weights(local_weights)
            self._network.load_state_dict(global_weights)
            if com % 1 == 0:
                # pdb.set_trace()
                test_acc = self._compute_accuracy(self._network, test_loader)
                info=("Task {}, Epoch {}/{} =>  Test_accy {:.2f}".format(
                    self._cur_task, com + 1, self.args["com_round"], test_acc,))
                prog_bar.set_description(info)
                if self.wandb == 1:
                    wandb.log({'Task_{}, accuracy'.format(self._cur_task): test_acc})


    # def _train(self, train_loader, test_loader):
    #     self._network.to(self._device)
    #     if self._old_network is not None:
    #         self._old_network.to(self._device)

    #     if self._cur_task == 0:
    #         optimizer = optim.SGD(
    #             self._network.parameters(),
    #             momentum=0.9,
    #             lr=init_lr,
    #             weight_decay=init_weight_decay,
    #         )
    #         scheduler = optim.lr_scheduler.MultiStepLR(
    #             optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
    #         )
    #         self._init_train(train_loader, test_loader, optimizer, scheduler)
    #     else:
    #         optimizer = optim.SGD(
    #             self._network.parameters(),
    #             lr=lrate,
    #             momentum=0.9,
    #             weight_decay=weight_decay,
    #         )
    #         scheduler = optim.lr_scheduler.MultiStepLR(
    #             optimizer=optimizer, milestones=milestones, gamma=lrate_decay
    #         )
    #         self._update_representation(train_loader, test_loader, optimizer, scheduler)

    # def _init_train(self, train_loader, test_loader, optimizer, scheduler):
    #     prog_bar = tqdm(range(init_epoch))
    #     for _, epoch in enumerate(prog_bar):
    #         self._network.train()
    #         losses = 0.0
    #         correct, total = 0, 0
    #         for i, (_, inputs, targets) in enumerate(train_loader):
    #             inputs, targets = inputs.to(self._device), targets.to(self._device)
    #             logits = self._network(inputs)["logits"]

    #             loss = F.cross_entropy(logits, targets)
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #             losses += loss.item()

    #             _, preds = torch.max(logits, dim=1)
    #             correct += preds.eq(targets.expand_as(preds)).cpu().sum()
    #             total += len(targets)

    #         scheduler.step()
    #         train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    #         if epoch % 5 == 0:
    #             info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
    #                 self._cur_task,
    #                 epoch + 1,
    #                 init_epoch,
    #                 losses / len(train_loader),
    #                 train_acc,
    #             )
    #         else:
    #             test_acc = self._compute_accuracy(self._network, test_loader)
    #             info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
    #                 self._cur_task,
    #                 epoch + 1,
    #                 init_epoch,
    #                 losses / len(train_loader),
    #                 train_acc,
    #                 test_acc,
    #             )
    #         prog_bar.set_description(info)

    #     print(info)

    # def _update_representation(self, train_loader, test_loader, optimizer, scheduler):

    #     prog_bar = tqdm(range(epochs))
    #     for _, epoch in enumerate(prog_bar):
    #         self._network.train()
    #         losses = 0.0
    #         correct, total = 0, 0
    #         for i, (_, inputs, targets) in enumerate(train_loader):
    #             inputs, targets = inputs.to(self._device), targets.to(self._device)
    #             logits = self._network(inputs)["logits"]    # last layer, 20

    #             fake_targets = targets - self._known_classes
    #             loss_clf = F.cross_entropy(
    #                 logits[:, self._known_classes :], fake_targets  # logits[10:20] -- 0~9 class
    #             )
    #             loss_kd = _KD_loss(
    #                 logits[:, : self._known_classes],   # logits on previous tasks
    #                 self._old_network(inputs)["logits"],
    #                 T,
    #             )

    #             loss = lamda * loss_kd + loss_clf

    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #             losses += loss.item()

    #             with torch.no_grad():
    #                 _, preds = torch.max(logits, dim=1)
    #                 correct += preds.eq(targets.expand_as(preds)).cpu().sum()
    #                 total += len(targets)

    #         scheduler.step()
    #         train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
    #         if epoch % 5 == 0:
    #             test_acc = self._compute_accuracy(self._network, test_loader)
    #             info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
    #                 self._cur_task,
    #                 epoch + 1,
    #                 epochs,
    #                 losses / len(train_loader),
    #                 train_acc,
    #                 test_acc,
    #             )
    #         else:
    #             info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
    #                 self._cur_task,
    #                 epoch + 1,
    #                 epochs,
    #                 losses / len(train_loader),
    #                 train_acc,
    #             )
    #         prog_bar.set_description(info)
    #     print(info)


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
