# cache_system.py
import sys
from collections import defaultdict

class CacheObject:
    def __init__(self, data, version: int, obj_type: str, source: str):
        """
        封装缓存对象信息
        :param data: 实际数据（模型参数、蒸馏样本、特征等）
        :param version: 版本号
        :param obj_type: 对象类型 ('model', 'distilled', 'feature')
        :param source: 来源层标识 ('end_1', 'edge_2', 'cloud')
        """
        self.data = data
        self.version = version
        self.type = obj_type
        self.source = source

class CacheLayer:
    def __init__(self, layer_name):
        self.layer_name = layer_name
        self.storage = {}           # key -> CacheObject
        self.version_vector = {}    # key -> version
        self.comm_cost_MB = 0.0
        self.hit = 0
        self.miss = 0

    def put(self, key, cache_object: CacheObject):
        self.storage[key] = cache_object
        self.version_vector[key] = cache_object.version

    def get(self, key):
        if key in self.storage:
            self.hit += 1
            return self.storage[key]
        else:
            self.miss += 1
            return None

    def diff_and_update(self, upper_layer, strategy="full", delta_v=0):
        """
        差量同步算法：
        strategy='full'：全量同步
        strategy='cvv-batch'：版本号差异 > delta_v 时批量同步
        strategy='cvv-event'：事件触发差量同步（此处可按需实现）
        """
        for key, upper_obj in upper_layer.storage.items():
            # 判定是否需要更新
            need_update = False
            if strategy == "full":
                need_update = True
            elif strategy in ["cvv-batch", "cvv-event"]:
                local_version = self.version_vector.get(key, -1)
                if upper_obj.version - local_version > delta_v:
                    need_update = True
            if need_update:
                self.storage[key] = upper_obj
                self.version_vector[key] = upper_obj.version
                size_MB = sys.getsizeof(upper_obj.data) / (1024*1024)
                self.comm_cost_MB += size_MB

    def reset_metrics(self):
        self.comm_cost_MB = 0.0
        self.hit = 0
        self.miss = 0

    def get_metrics(self, total_requests):
        chr = self.hit / max(total_requests, 1)
        return chr, self.comm_cost_MB
