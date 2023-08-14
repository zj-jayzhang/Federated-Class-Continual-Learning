import numpy as np
from tqdm import tqdm
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.data_manager import partition_data, DatasetSplit, average_weights, setup_seed
import copy, wandb

EPSILON = 1e-8
T = 2




def print_data_stats(client_id, train_data_loader):
    # pdb.set_trace()
    def sum_dict(a,b):
        temp = dict()
        for key in a.keys() | b.keys():
            temp[key] = sum([d.get(key, 0) for d in (a, b)])
        return temp
    temp = dict()
    for batch_idx, (_, images, labels) in enumerate(train_data_loader):
        unq, unq_cnt = np.unique(labels, return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        temp = sum_dict(tmp, temp)
    return sorted(temp.items(),key=lambda x:x[0])



class iCaRL(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self.memory_size = args["memory_size"]

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes

    def get_all_previous_dataset(self, data_manager, idx):
        # for second task, self._cur_task=1
        bgn_cls, end_cls = 0, self.each_task
        train_dataset = data_manager.get_dataset(
            np.arange(bgn_cls, end_cls),
            source="train",
            mode="train",
        )
        setup_seed(self.seed)
        user_groups = partition_data(train_dataset.labels, beta=self.args["beta"], n_parties=self.args["num_users"])
        all_previous_dataset = DatasetSplit(train_dataset, user_groups[idx])
        # for third task
        for i in range(2, self._cur_task+1):  # 2-4
            setup_seed(self.seed)
            bgn_cls += self.each_task  # 20-40
            end_cls += self.each_task
            train_dataset_next = data_manager.get_dataset(
                np.arange(bgn_cls, end_cls),
                source="train",
                mode="train",
            )
            user_groups_next = partition_data(train_dataset_next.labels, beta=self.args["beta"], n_parties=self.args["num_users"])
            tmp_dataset = DatasetSplit(train_dataset_next, user_groups_next[idx])  # <utils.data_manager.DummyDataset>
            all_previous_dataset = self.combine_dataset(all_previous_dataset, tmp_dataset, 0) # combine two datasets
            all_previous_dataset = DatasetSplit(all_previous_dataset, range(all_previous_dataset.labels.shape[0]))
            # 2417->   all_previous_dataset.idxs[0:4]= [9013, 7479, 5185, 7241]
        return all_previous_dataset





    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        print("Learning on {}-{}".format(self._known_classes, self._total_classes))
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),   # get memory, 2000 data: 100 * 20cls[0~19]
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=256, shuffle=False, num_workers=4
        )
        self._network.cuda()
        setup_seed(self.seed)
        self._fl_train(train_dataset, self.test_loader, data_manager)


    def _local_update(self, model, train_data_loader, client_id, tmp, com_id):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        for iter in range(self.args["local_ep"]):
            total = 0
            for batch_idx, (_, images, labels) in enumerate(train_data_loader):
                images, labels = images.cuda(), labels.cuda()
                output = model(images)["logits"]
                loss = F.cross_entropy(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += images.shape[0]
            if iter ==0 and com_id==0 : print("task_id:{}, client_id: {}, local dataset size: {}, labels:{}".format(self._cur_task ,client_id, total, tmp))
        return model.state_dict()

    def _local_finetune(self, model, train_data_loader, client_id, tmp, com_id):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    
        for iter in range(self.args["local_ep"]):
            total = 0
            for batch_idx, (_, images, labels) in enumerate(train_data_loader):
                images, labels = images.cuda(), labels.cuda()
                # fake_targets = labels - self._known_classes
                output = model(images)["logits"]
                #* finetune on the new tasks
                loss_clf = F.cross_entropy(output, labels)
                loss_kd = _KD_loss(
                    output[:, : self._known_classes],
                    self._old_network(images)["logits"],
                    T,
                )

                loss = loss_clf + loss_kd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += images.shape[0]
            if iter ==0 and com_id==0 : print("task_id:{}, client_id: {}, local dataset size: {}, labels:{}".format(self._cur_task ,client_id, total, tmp))
        return model.state_dict()

    def _fl_train(self, train_dataset, test_loader, data_manager):
        self._network.cuda()
        user_groups = partition_data(train_dataset.labels, beta=self.args["beta"], n_parties=self.args["num_users"])
        prog_bar = tqdm(range(self.args["com_round"]))
        
        for _, com in enumerate(prog_bar):
            local_weights = []
            m = max(int(self.args["frac"] * self.args["num_users"]), 1)
            idxs_users = np.random.choice(range(self.args["num_users"]), m, replace=False)
            for idx in idxs_users:
                # update local train data
                if self._cur_task == 0:
                    local_dataset = DatasetSplit(train_dataset, user_groups[idx])
                else:

                    current_local_dataset = DatasetSplit(train_dataset, user_groups[idx])
                    previous_local_dataset = self.get_all_previous_dataset(data_manager, idx) 

                    local_dataset = self.combine_dataset(previous_local_dataset, current_local_dataset, self.memory_size)
                    local_dataset = DatasetSplit(local_dataset, range(local_dataset.labels.shape[0]))

                local_train_loader = DataLoader(local_dataset, batch_size=self.args["local_bs"], shuffle=True, num_workers=4)
                tmp = print_data_stats(idx, local_train_loader)
                if com !=0:
                    tmp = ""
                if self._cur_task == 0:                    
                    w = self._local_update(copy.deepcopy(self._network), local_train_loader, idx, tmp, com)
                else:
                    w = self._local_finetune(copy.deepcopy(self._network), local_train_loader, idx, tmp, com)
                local_weights.append(copy.deepcopy(w))
            # update global weights
            global_weights = average_weights(local_weights)
            self._network.load_state_dict(global_weights)
            if com % 1 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info=("Task {}, Epoch {}/{} =>  Test_accy {:.2f}".format(
                    self._cur_task, com + 1, self.args["com_round"], test_acc,))
                prog_bar.set_description(info)
                if self.wandb == 1:
                    wandb.log({'Task_{}, accuracy'.format(self._cur_task): test_acc})







def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
