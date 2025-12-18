import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.utils import save_image
import torchvision.transforms as transforms
import utils
import torch
from cache_system import CacheLayer, CacheObject
from utils import accuracy
from autoencoder_pretrained import create_autoencoder
from torch import nn
from utils import KL_Loss, CE_Loss

# === 新增 ===
from dataclasses import dataclass, field
from typing import List, Dict, Any

# === 新增：指标容器 ===
@dataclass
class FedAggMetrics:
    acc_list: List[float] = field(default_factory=list)
    consistency_list: List[float] = field(default_factory=list)
    comm_cost_cloud_MB: List[float] = field(default_factory=list)
    comm_cost_edges_MB: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "acc_list": self.acc_list,
            "consistency_list": self.consistency_list,
            "comm_cost_cloud_MB": self.comm_cost_cloud_MB,
            "comm_cost_edges_MB": self.comm_cost_edges_MB,
        }

# === 新增：估算模型尺寸（MB）用于通信量近似 ===
def model_size_mb(model: torch.nn.Module) -> float:
    total_params = 0
    for p in model.parameters():
        total_params += p.numel()
    return total_params * 4.0 / (1024.0 * 1024.0)

# === 新增：Top-1 准确率（云端模型在测试集）===
@torch.no_grad()
def eval_top1_acc(model: torch.nn.Module, test_loader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in test_loader:
        x = x.to(device)
        y = torch.as_tensor(y, dtype=torch.long, device=device)
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)

# === 新增：一致性（边-云 logits 差异；越接近 1 越一致）===
@torch.no_grad()
def compute_consistency_logits(cloud_model: torch.nn.Module,
                               edge_models: list,
                               test_loader,
                               device: torch.device,
                               max_batches: int = 1) -> float:
    """
    使用测试数据前 max_batches 个 batch 进行云-边 logits 差异一致性度量。
    修正点：在评估前将云、边模型统一迁移到相同 device，避免 CPU/GPU 混用。
    返回值越接近 1 一致性越好。
    """
    if not edge_models:
        return 1.0

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 统一设备
    cloud_model = cloud_model.to(device)
    for m in edge_models:
        m.to(device)

    cloud_model.eval()
    for m in edge_models:
        m.eval()

    batch_cnt = 0
    dists = []
    for x, _ in test_loader:
        x = x.to(device, non_blocking=True)
        out_c = cloud_model(x)
        if isinstance(out_c, tuple):
            out_c = out_c[0]
        out_c = out_c.detach().float()

        for em in edge_models:
            out_e = em(x)
            if isinstance(out_e, tuple):
                out_e = out_e[0]
            out_e = out_e.detach().float()
            # 每样本 L2 后取均值
            l2 = torch.linalg.vector_norm(out_c - out_e, dim=1).mean().item()
            dists.append(l2)

        batch_cnt += 1
        if batch_cnt >= max_batches:
            break

    if len(dists) == 0:
        return 1.0
    avg_l2 = float(sum(dists) / len(dists))
    return 1.0 / (1.0 + avg_l2)


def test_on_cloud(cloud_model,test_data_global,comm_round):
    acc_all=[]
    if True:
        loss_avg = utils.RunningAverage()
        accTop1_avg = utils.RunningAverage()
        accTop5_avg = utils.RunningAverage()
        for batch_idx, (images, labels) in enumerate(test_data_global):
            images, labels = images.cuda(), labels.cuda()
            labels=torch.tensor(labels,dtype=torch.long)
            log_probs, extracted_features = cloud_model(images)
            metrics = utils.accuracy(log_probs, labels, topk=(1, 5))
            accTop1_avg.update(metrics[0].item())
            accTop5_avg.update(metrics[1].item())
        test_metrics = {
                        'test_accTop1': accTop1_avg.value(),
                        'test_accTop5': accTop5_avg.value(),
                        }
        print("Test/AccTop1 in comm_round",comm_round,test_metrics['test_accTop1'])


def run_fedagg(client_models, edge_models, cloud_model,
               train_data_local_num_dict, test_data_local_num_dict,
               train_data_local_dict, test_data_local_dict,
               test_data_global, args):

    V1=[Node(args,cloud_model)]
    V2=create_child_for_upper_level(args,V1,args.edge_number,edge_models)
    # === 初始化缓存层 ===
    cloud_cache = CacheLayer("cloud")
    edge_caches = [CacheLayer(f"edge_{i}") for i in range(args.edge_number)]
    end_caches = [CacheLayer(f"end_{i}") for i in range(args.client_number)]

    metrics = {
        "acc_list": [],
        "consistency_list": [],
        "comm_cost_cloud_MB": [],
        "comm_cost_edges_MB": [],
        "chr_cloud": [],
        "chr_edges": [],
        "chr_ends": []
    }

    # === 初始化 CVV 参数 ===
    sync_strategy = getattr(args, "sync_strategy", "full")
    T_sync = getattr(args, "T_sync", 1)
    delta_v = getattr(args, "delta_v", 0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for r in range(args.comm_round):
        # === 原有训练聚合逻辑 ===
        train_FedAgg(V1[0],args)
        test_on_cloud(V1[0].model,test_data_global,r)
        # 每轮更新 end_caches
        for end_idx, model in enumerate(client_models):
            obj = CacheObject(data=model.state_dict(),
                              version=r,
                              obj_type="model",
                              source=f"end_{end_idx}")
            end_caches[end_idx].put("model", obj)

        # 端→边 差量同步
        for e_idx in range(args.edge_number):
            for end_id in range(e_idx*(args.client_number//args.edge_number),
                                (e_idx+1)*(args.client_number//args.edge_number)):
                edge_caches[e_idx].diff_and_update(end_caches[end_id],
                                                   strategy=sync_strategy,
                                                   delta_v=delta_v)

        # 边→云 差量同步
        for e_idx in range(args.edge_number):
            cloud_cache.diff_and_update(edge_caches[e_idx],
                                        strategy=sync_strategy,
                                        delta_v=delta_v)

        # === 记录指标 ===
        # 计算云端模型精度
        acc_val = eval_top1_acc(V1[0].model, test_data_global, device)
        metrics["acc_list"].append(acc_val)
        # 一致性 (简化为云-边参数版本一致率)
        #cons_val = consistency_version(cloud_cache, edge_caches)
        cons_val = compute_consistency_logits(V1[0].model, edge_models_rt, test_data_global, device, max_batches=1)
        metrics["consistency_list"].append(cons_val)
        # 通信开销
        metrics["comm_cost_cloud_MB"].append(cloud_cache.comm_cost_MB)
        metrics["comm_cost_edges_MB"].append(sum([ec.comm_cost_MB for ec in edge_caches]))
        # 命中率
        chr_c, _ = cloud_cache.get_metrics(total_requests=len(edge_caches))
        metrics["chr_cloud"].append(chr_c)
        metrics["chr_edges"].append(sum([ec.get_metrics(1)[0] for ec in edge_caches]) / args.edge_number)
        metrics["chr_ends"].append(sum([ec.get_metrics(1)[0] for ec in end_caches]) / args.client_number)

        print("round:", r, " acc:", acc_val, " consistency:", cons_val, " comm_cost_cloud_MB:", cloud_cache.comm_cost_MB, " comm_cost_edges_MB:", sum([ec.comm_cost_MB for ec in edge_caches]), " total_cache:", cloud_cache.comm_cost_MB+sum([ec.comm_cost_MB for ec in edge_caches]), " chr_cloud:", chr_c, " chr_edges:", sum([ec.get_metrics(1)[0] for ec in edge_caches]) / args.edge_number, " chr_ends:", sum([ec.get_metrics(1)[0] for ec in end_caches]) / args.client_number)

        # 重置本轮通信统计
        cloud_cache.reset_metrics()
        for ec in edge_caches: ec.reset_metrics()
# === 新增：返回字典，供上层写 CSV ===
    return metrics

@torch.no_grad()
def eval_top1_acc(model: torch.nn.Module, test_loader, device: torch.device) -> float:
    """
    评估前，将模型移动到 device；将每个 batch 的数据也移动到相同 device。
    不依赖外部状态，避免 CPU/GPU 混用导致的 conv 错误。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 确保模型在 device 上
    model = model.to(device)
    model.eval()

    correct, total = 0, 0
    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y, dtype=torch.long, device=device)
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def consistency_version(cloud_cache, edge_caches):
    """
    一致性度量：云端与各边节点相同key的版本号一致率
    """
    total_keys = 0
    match_keys = 0
    for ec in edge_caches:
        for key, ver in ec.version_vector.items():
            total_keys += 1
            if cloud_cache.version_vector.get(key, None) == ver:
                match_keys += 1
    return match_keys / max(total_keys, 1)

global_index=0
class Node:
    def __init__(self, args, model, father=None):
         global global_index
         self.model=model.cuda()
         self.dataset=None
         self.father = father
         self.children = []
         self.index=global_index
         self.autoencoder=create_autoencoder().cuda()
         self.noises=[]
         self.labels=[]
         self.args=args
         global_index=global_index+1
    
    def is_leaf(self):
        return len(self.children)==0

    def is_root(self):
        return self.father==None

    def __repr__(self):
        return '[index:'+str(self.index)+' data:'+str(self.data)+' father:'+str(self.father.index)+\
            ' +children:'+(str([_.index for _ in self.children]) if len(self.children)!=0 else "[]")+']'

def create_child_for_upper_level(args,upper_level,children_number,models):
    result=[]
    for ele in upper_level:
        sub_nodes=[]
        for _,model in zip(range(children_number),models):
            sub_nodes.append(Node(args,model,ele))
        ele.children=sub_nodes
        result.extend(sub_nodes)
    return result


def train_FedAgg(node,args):
    if node.is_root():
        for child in node.children:
            train_FedAgg(child,args)
    elif node.is_leaf():
        BSBODP(node,node.father,args)
    else:
        for child in node.children:
            train_FedAgg(child,args)
        BSBODP(node,node.father,args)

def Init(node):
    if node.is_root():
        for child in node.children:
            Init(child)
    elif node.is_leaf():
        for idx,data in enumerate(node.dataset):
            img,label=data
            img,label=img.cuda(),label.cuda()
            noise=node.autoencoder.encoder(img)
            node.noises.append(noise)
            node.labels.append(label)
        node.father.noises.extend(node.noises)
        node.father.labels.extend(node.labels)
    else:
        for child in node.children:
            Init(child)
        node.father.noises.extend(node.noises)
        node.father.labels.extend(node.labels)
        
def BSBODP(node1,node2,args):
    BSBODP_dir(node1,node2,args)
    BSBODP_dir(node2,node1,args)


class Loss_Non_Leaf(nn.Module):
    def __init__(self, temperature=1, alpha=10):
        super(Loss_Non_Leaf, self).__init__()
        self.alpha = alpha
        self.kl_loss_crit=KL_Loss(temperature)
        self.ce_loss_crit=nn.CrossEntropyLoss()

    def forward(self, output_batch, teacher_outputs, label):
        
        loss_ce=self.ce_loss_crit(output_batch,label.long())
        loss_kl=self.kl_loss_crit(output_batch,teacher_outputs.detach())
        loss_true=loss_ce+self.alpha*loss_kl
        return loss_true


class Loss_Leaf(nn.Module):
    def __init__(self, temperature=1, alpha=1, alpha2=1):
        super(Loss_Leaf, self).__init__()
        self.alpha = alpha
        self.alpha2 = alpha2
        self.non_leaf_loss_crit=Loss_Non_Leaf(temperature,alpha)
        self.ce_loss_crit=nn.CrossEntropyLoss()
    def forward(self, output_fake, teacher_outputs_fake, output_true, label):
        loss_leaf=self.non_leaf_loss_crit(output_fake, teacher_outputs_fake.detach(), label.long())
        loss_ce=self.ce_loss_crit(output_true,label.long())
        loss_true=loss_leaf+self.alpha2*loss_ce
        return loss_true


def BSBODP_dir(node_origin,node_neigh,args):
    noises=node_neigh.noises if len(node_neigh.noises)<len(node_origin.noises) else node_origin.noises
    labels=node_neigh.labels if len(node_neigh.labels)<len(node_origin.labels) else node_origin.labels
    # 安全获取温度参数，避免 args 无 T_agg 报错
    temp = getattr(args, "T_agg", 3.0)
    crit_non_leaf=Loss_Non_Leaf(args.T_agg)
    crit_leaf=Loss_Leaf(args.T_agg)
    optimizer = torch.optim.SGD(node_origin.model.parameters(), lr=node_origin.args.lr, momentum=0.9)
    for idx,(noise,label) in enumerate(zip(noises,labels)):
        optimizer.zero_grad()
        fake_data=node_neigh.autoencoder.decoder(noise)
        nei_logits,_=node_neigh.model(fake_data)
        logits_fake,_=node_origin.model(fake_data)
        loss=0.0
        if node_origin.is_leaf():
            img,label_=node_origin.dataset[idx]
            img,label_=img.cuda(),label_.cuda()
            assert(label,label_)
            logits_true,_=node_origin.model(img)
            loss=loss+crit_leaf(logits_fake,nei_logits,logits_true,label.long())
        else:
            loss=loss+crit_non_leaf(logits_fake,nei_logits,label.long())
        loss.backward(retain_graph=True)
        optimizer.step()