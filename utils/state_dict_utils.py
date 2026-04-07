from collections import OrderedDict
import warnings
import torch


def remove_prefix(state_dict, prefix):
    n_prefix_parts = len(prefix.split('.'))
    new_dict = OrderedDict()
    for k, v in state_dict.items():
        k_parts = k.split('.')
        if k.startswith(prefix):
            for i in range(n_prefix_parts):
                k_parts.pop(0)
            new_k = '.'.join(k_parts)
            new_dict[new_k] = v

    return new_dict


def add_prefix(state_dict, prefix):
    if not prefix.endswith('.'):
        prefix += '.'
    new_dict = OrderedDict()
    for k, v in state_dict.items():
        new_dict[prefix + k] = v

    return new_dict


def merge_state_dicts(state_dicts, strict=True):
    """
    将嵌套的state_dict展平为一个扁平的OrderedDict
    
    Args:
        state_dicts: 可能是多级嵌套的dict或OrderedDict
        strict: 是否严格检查嵌套字典的键名衔接
        
    Returns:
        OrderedDict: 展平后的state_dict，其中key是按照嵌套层级用点号连接的
    """
    result = OrderedDict()
    
    def _merge(d, prefix=''):
        for k, v in d.items():
            if isinstance(v, dict) or isinstance(v, OrderedDict):
                # 如果值是字典，递归处理
                key_parts = k.split('.')
                key_sub_name = key_parts[-1]
                # key_main_name = '.'.join(key_parts[:-1] if len(key_parts) > 1 else key_parts)
                key_main_name = '.'.join(key_parts[:-1])
                value_sub_name_coincedence = {vk: vk.split('.')[0] == key_sub_name for vk in v.keys()}
                if strict:
                    assert all(value_sub_name_coincedence.values()), f"嵌套字典的键名不匹配: {value_sub_name_coincedence}"
                    new_prefix = f"{prefix}{'.' if key_main_name else ''}{key_main_name}" if prefix else key_main_name
                else:
                    if all(value_sub_name_coincedence.values()):
                        new_prefix = f"{prefix}{'.' if key_main_name else ''}{key_main_name}" if prefix else key_main_name
                    else:
                        warnings.warn(f"嵌套字典的键名不匹配: {value_sub_name_coincedence}, 将采用直接拼接")
                        new_prefix = f"{prefix}.{k}" if prefix else k
                _merge(v, new_prefix)
            elif isinstance(v, torch.Tensor):
                # 如果值是Tensor，直接添加到结果中
                new_key = f"{prefix}.{k}" if prefix else k
                result[new_key] = v
            else:
                raise ValueError(f"不支持嵌套字典的键值类型: {type(v)}")
    
    _merge(state_dicts)
    return result


def extract_state_dict_with_prefix(state_dict, prefix, add_prefix_last=True):
    prefix_parts = prefix.split('.')
    new_dict = OrderedDict()
    for k, v in state_dict.items():
        k_parts = k.split('.')
        if k.startswith(prefix):
            for i in range(len(prefix_parts)):
                k_parts.pop(0)
            new_k = '.'.join(k_parts)
            new_dict[new_k] = v

    if add_prefix_last:
        new_dict = add_prefix(new_dict, prefix_parts[-1])

    return new_dict


def deep_copy_state_dict(original_state_dict):
    new_state_dict = OrderedDict()
    for key, tensor in original_state_dict.items():
        # 关键操作：创建内容相同但内存独立的新Tensor
        new_state_dict[key] = tensor.detach().clone()
    return new_state_dict


if __name__ == '__main__':
    from collections import OrderedDict
    a = OrderedDict()
    a['model.weight'] = torch.Tensor(0)
    a['model.bias'] = torch.Tensor(0)
    b = OrderedDict()
    b['encoder.weight'] = torch.Tensor(1)
    b['encoder.bias'] = torch.Tensor(1)
    c = OrderedDict()
    c['decoder.weight'] = torch.Tensor(2)
    c['decoder.bias'] = torch.Tensor(2)
    d = OrderedDict()
    d['head.weight'] = torch.Tensor(3)
    d['head.bias'] = torch.Tensor(3)

    c['decoder.head'] = d
    a['model.encoder'] = b
    a['model.decoder'] = c

    state_dict = merge_state_dicts(a)
    state_dict2 = remove_prefix(state_dict, 'model')
    state_dict3 = add_prefix(state_dict2, 'new_model')
    state_dict4 = extract_state_dict_with_prefix(state_dict3, 'new_model.encoder')

    ii = 0