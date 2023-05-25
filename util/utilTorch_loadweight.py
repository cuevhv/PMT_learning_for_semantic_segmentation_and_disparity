import torch
import os
import shutil


def load_checkpoint_and_params(CFG, rank, load_weights_by_name, scaler, net, optimizer):
    start_e = 0
    best_metric = [1, 0]
    epoch_history = []
    IoU_history_val = []
    disp_history_val = []
    loss_history_val = []
    IoU_history_train = []
    disp_history_train = []
    loss_history_train = []

    if CFG.f16:
        import apex
    if CFG.load_weights:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        if CFG.load_weights.rsplit('.', 1)[-1] == 'tar':
            checkpoint = torch.load(CFG.load_weights, map_location=map_location if CFG.nodes else 'cpu')

            state_dict = checkpoint['state_dict']
            # del state_dict["module.resnet_features.resnet_features.classifier.weight"]
            # del state_dict["module.resnet_features.resnet_features.classifier.bias"]
            # model_dict = list(net.state_dict().keys())
            # state_dict = changeWeightName(state_dict, model_dict)

            if load_weights_by_name:
                print('loading weights by name')
                own_state = net.state_dict()
                for name, param in state_dict.items():
                    if name == "module.Conv2DownUp11.1.ct2d.weight":
                        if "module.convOutput.ct2d.weight" in own_state:
                            print('found')
                            param = param.data
                            own_state["module.convOutput.ct2d.weight"].copy_(param)
                    if name not in own_state or name == 'module.segNet.Conv2DownUp2.1.ct2d.weight' or \
                            name == 'module.Conv2DownUp11.1.ct2d.weight':
                        continue
                    # if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                    own_state[name].copy_(param)
                net.load_state_dict(own_state)
            else:
                net.load_state_dict(state_dict)

            if CFG.f16 and 'amp' in checkpoint.keys():
                apex.amp.load_state_dict(checkpoint['amp'])
                print('amp dict loaded')
            else:
                print('no amp dict')

            if CFG.torch_amp and 'amp' in checkpoint.keys():
                scaler.load_state_dict(checkpoint['amp'])
                print('torch amp dict loaded')
            else:
                print('no cuda amp')

            start_e = checkpoint['epoch']
            if rank == 0:
                print('loading weights')
                print('training will start from epoch: ', start_e)

            if not load_weights_by_name:
                optimizer_dict = checkpoint['optimizer']
                optimizer.load_state_dict(optimizer_dict)

            if 'IoU_history_val' in checkpoint.keys():
                best_metric = checkpoint['best_metric']
                epoch_history = checkpoint['epoch_history']
                IoU_history_val = checkpoint['IoU_history_val']
                disp_history_val = checkpoint['disp_history_val']
                loss_history_val = checkpoint['loss_history_val']
                IoU_history_train = checkpoint['IoU_history_train']
                disp_history_train = checkpoint['disp_history_train']
                loss_history_train = checkpoint['loss_history_train']
        else:
            # from collections import OrderedDict
            # new_state_dict = OrderedDict()
            # for k, v in torch.load(CFG.load_weights).items():
            #     name = "module."+k#k.replace(".module", '') # remove 'module.' of dataparallel
            #     new_state_dict[name]=v

            # net.load_state_dict(new_state_dict)
            if load_weights_by_name:
                print('loading weights by name')
                own_state = net.state_dict()
                for name, param in torch.load(CFG.load_weights).items():
                    if name == "module.Conv2DownUp11.1.ct2d.weight":
                        if "module.convOutput.ct2d.weight" in own_state:
                            print('found')
                            param = param.data
                            own_state["module.convOutput.ct2d.weight"].copy_(param)
                    if name not in own_state:
                        continue
                    # else:
                    #     print('Freezing {}'.format(name))
                    #     param.requires_grad_(False)
                    # if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                    own_state[name].copy_(param)
                net.load_state_dict(own_state)
            else:
                net.load_state_dict(torch.load(CFG.load_weights))
            start_e = 0
    else:
        start_e = 0

    print(start_e, 'best metric', best_metric)
    return start_e, best_metric, epoch_history, IoU_history_val, disp_history_val, loss_history_val, \
        IoU_history_train, disp_history_train, loss_history_train
