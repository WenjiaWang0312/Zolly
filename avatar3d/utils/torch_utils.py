import torch
import numpy as np


def unbind_pose(body_or_hand_pose):
    batch_size = body_or_hand_pose.shape[0]
    body_or_hand_pose = body_or_hand_pose.reshape(batch_size, -1, 3)
    pose_list = torch.unbind(body_or_hand_pose, 1)
    return pose_list


def merge_dict(*dicts):
    result = dict()
    for dic in dicts:
        for k in dic:
            result[k] = dic[k]
    return result


def cat_pose_list(pose_list, dim=1):
    if isinstance(pose_list, torch.Tensor):
        return pose_list
    else:
        return torch.cat(pose_list, dim)


def cat(list_of_tensor_or_array, dim):
    if isinstance(list_of_tensor_or_array[0], torch.Tensor):
        return torch.cat(list_of_tensor_or_array, dim)
    else:
        return np.concatenate(list_of_tensor_or_array, dim)


def build_parameters(**kwargs):
    batch_size = kwargs.pop('batch_size', 1)
    parameters = {}
    for k in kwargs:
        if isinstance(kwargs[k], dict):
            shape = list(kwargs[k]['shape'])
            if shape[0] == -1:
                shape[0] = batch_size
            value = kwargs[k]['value']
            parameters[k] = torch.nn.parameter.Parameter(torch.ones(shape) *
                                                         value,
                                                         requires_grad=False)
        else:

            shape = list(kwargs[k])
            if shape[0] == -1:
                shape[0] = batch_size
            parameters[k] = torch.nn.parameter.Parameter(torch.zeros(shape),
                                                         requires_grad=False)

    return parameters


def dict2tensor(input_dict, device=None, use_float=True):
    for k, v in input_dict.items():
        if isinstance(v, np.ndarray):
            try:
                if use_float:
                    input_dict[k] = torch.tensor(v).float()
                else:
                    input_dict[k] = torch.tensor(v)
                if device is not None:
                    input_dict[k] = input_dict[k].to(device)
            except:
                pass
        elif isinstance(v, torch.Tensor):
            if use_float:
                input_dict[k] = v.float()
            if device is not None:
                input_dict[k] = v.to(device)
        elif isinstance(v, dict):
            input_dict[k] = dict2tensor(v, device, use_float)

    return input_dict


def convert_RGB_BGR(image, axis=-1):
    if isinstance(image, torch.Tensor):
        splits = list(torch.split(image, 1, axis))
        temp = splits[2].clone()
        splits[2] = splits[0].clone()
        splits[0] = temp
        return torch.cat(splits, -1)
    elif isinstance(image, np.ndarray):
        splits = list(np.split(image, image.shape[-1], axis))
        temp = splits[2].copy()
        splits[2] = splits[0].copy()
        splits[0] = temp
        return np.concatenate(splits, -1)


def dict2numpy(input_dict):
    for k, v in input_dict.items():
        if isinstance(v, torch.Tensor):
            input_dict[k] = v.detach().cpu().numpy()
        elif isinstance(v, dict):
            input_dict[k] = dict2numpy(v)
        elif isinstance(v, (tuple, list)):
            for v_ in v:
                if isinstance(v_, (tuple, list)):
                    v = [v_.detach().cpu().numpy() for v_ in v]
    return input_dict


def to_tensor(array_or_tensor, device=None):
    if array_or_tensor is None:
        return array_or_tensor
    else:
        if isinstance(array_or_tensor, np.ndarray):
            array_or_tensor = torch.tensor(array_or_tensor)
        elif isinstance(array_or_tensor, torch.Tensor):
            array_or_tensor = array_or_tensor
        if device is not None:
            array_or_tensor = array_or_tensor.to(device)
        return array_or_tensor


def image_tensor2numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, torch.Tensor):
        tensor_ = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        array = tensor_.detach().cpu().numpy()
        array = (array * 255).astype(np.uint8)
        return array


def move_dict_to_device(input_dict, device):
    for k, v in input_dict.items():
        if isinstance(v, torch.Tensor):
            input_dict[k] = v.to(device).float()
        elif isinstance(v, dict):
            move_dict_to_device(v, device)
        elif isinstance(v, (tuple, list)):
            try:
                input_dict[k] = [v_.to(device) for v_ in v]
            except:
                pass


def slice_dict(input_dict, index, reserved_keys=()):
    out_dict = dict()
    if isinstance(index, (np.ndarray, torch.Tensor)):
        index = index.tolist()
    for k, v in input_dict.items():
        if hasattr(v, '__getitem__') and not isinstance(
                v, str) and not k in reserved_keys and not (isinstance(
                    v, dict)):
            if isinstance(v, (torch.Tensor, np.ndarray)) and not v.shape:
                out_dict[k] = v

            elif len(v) == 1:
                out_dict[k] = v
            elif isinstance(v, (list, tuple)):
                out_dict[k] = [v[i] for i in index]
            else:
                out_dict[k] = v[index]

        elif isinstance(v, dict):
            out_dict[k] = slice_dict(v, index)
        else:
            out_dict[k] = v
    return out_dict


def slice_pose_dict(input_dict, index):
    out_dict = dict()
    if index is None:
        return input_dict
    else:
        index = index.long() if isinstance(index, torch.Tensor) else index
        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                if len(v) == 1:
                    out_dict[k] = v
                else:
                    out_dict[k] = v[index]
            elif isinstance(v, (tuple, list)):
                out_dict[k] = cat(v, 1)[index]
        return out_dict


def concat_dict_list(list_of_dict):
    output = dict()
    keys = list_of_dict[0].keys()
    for k in keys:
        if isinstance(list_of_dict[0][k], torch.Tensor):
            selected_tensors = [curr_dict[k] for curr_dict in list_of_dict]
            output[k] = cat(selected_tensors, 0)
        elif isinstance(list_of_dict[0][k], np.ndarray):
            selected_tensors = [curr_dict[k] for curr_dict in list_of_dict]
            output[k] = cat(selected_tensors, 0)
        elif isinstance(list_of_dict[0][k], dict):
            output[k] = concat_dict_list(
                [curr_dict[k] for curr_dict in list_of_dict])
    return output


def load_state_dict(model, state_dict, strict=False):
    if isinstance(state_dict, str):
        state = torch.load(state_dict, map_location='cpu')
    elif isinstance(state_dict, dict):
        state = state_dict
    else:
        raise ValueError('type of state_dict is illegal')

    if strict:
        model.load_state_dict(state, strict=True)
    else:
        if 'state_dict' in state:
            state = state['state_dict']

        state_dict = {}
        for k, v in state.items():
            if k.startswith('module.'):
                state_dict[k[len('module.'):]] = v
            else:
                state_dict[k] = v

        new_state_dict = {}
        for k, v in model.state_dict().items():
            if k in state_dict and v.size() == state_dict[k].size():
                new_state_dict[k] = state_dict[k]

        model.load_state_dict(new_state_dict, strict=False)

        print('Finish loading state: {}/{}'.format(len(new_state_dict),
                                                   len(model.state_dict())))

    return model


def merge_loss_dict(*args):
    keys = []
    losses = dict()
    total_loss = 0
    for loss_dict in args:
        keys += list(loss_dict.keys())
    for k in keys:
        losses[k] = 0.
        for loss_dict in args:
            loss = loss_dict.get(k, 0.)
            if isinstance(loss, torch.Tensor):
                if loss.ndim == 3:
                    total_loss += loss.sum(dim=(2, 1))
                elif loss.ndim == 2:
                    total_loss += loss.sum(dim=-1)
                else:
                    total_loss += loss
            losses[k] += loss
    losses['total_loss'] = total_loss
    return losses
