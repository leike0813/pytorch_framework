import warnings
from ast import literal_eval
import torch
import torch.nn.functional as F
from torch.nn import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, SmoothL1Loss, HuberLoss, KLDivLoss
from ..utils.invisible_yacs import CustomCfgNode as CN
from .custom_losses import (
    DiceLoss,
    GeneralizedDiceLoss,
    BinaryFocalLoss,
    FocalLoss,
    BinaryFocalWithLogitsLoss,
    GeneralizedFocalLoss,
    GeneralizedBinaryFocalLoss,
    SmoothL1NormalizedBCELoss,
    SmoothL1NormalizedBCEWithLogitsLoss,
    OrderedCrossEntropyLoss,
    PositionLoss,
)
from .default_config import DEFAULT_CONFIG


def build_lossfunc(node=DEFAULT_CONFIG):
    loss_fn, loss_hparams = get_lossfunc(node)
    loss_weight = node.WEIGHT
    sideoutput_weights = node.SIDE_OUTPUT_WEIGHTS
    num_sideoutput = len(sideoutput_weights)

    def loss_func_with_sideoutput(y_pred, y, *args, **kwargs):
        loss = loss_weight * sum([
            sideoutput_weights[i]
            * loss_fn(y_pred[i], y, *args, **kwargs)
            for i in range(num_sideoutput)
        ])
        return loss

    def loss_func(y_pred, y, *args, **kwargs):
        loss = loss_weight * loss_fn(y_pred[0] if num_sideoutput > 1 else y_pred , y, *args, **kwargs)
        return loss

    return loss_func, loss_func_with_sideoutput, loss_hparams


def get_lossfunc(node, func_name=None, append_funcname=True):
    assert node.OUTPUT_TYPE in ['logit', 'prob', 'logprob', 'real'], 'Invalid model output type: {}'.format(node.OUTPUT_TYPE)
    def node_name():
        return '_' + node.NAME if node.NAME != '' else ''

    if func_name:
        func_name = func_name
    else:
        func_name = node.FUNC_NAME
    hparams = {'loss-function{nme}'.format(nme=node_name()): func_name} if append_funcname else {}
    hparams.update({
        'loss-weight{nme}'.format(nme=node_name()): node.WEIGHT,
        'side-output-weights{nme}'.format(nme=node_name()): node.SIDE_OUTPUT_WEIGHTS,
        # 'supervision-types{nme}'.format(nme=node_name()): node.SIDE_OUTPUT_SUPERVISION_TYPES,
        'model-output-type{nme}'.format(nme=node_name()): node.OUTPUT_TYPE,
    })

    if func_name == 'BCELoss':
        device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda") # Caution! This line is not of general purpose
        if node.OUTPUT_TYPE == 'prob':
            loss_func = BCELoss(reduction=node.BCELOSS.reduction, pos_weight=torch.Tensor(node.BCELOSS.pos_weight).to(device), **node.CUSTOM_ARGS)
        elif node.OUTPUT_TYPE == 'logit':
            loss_func = BCEWithLogitsLoss(reduction=node.BCELOSS.reduction, pos_weight=torch.Tensor(node.BCELOSS.pos_weight).to(device), **node.CUSTOM_ARGS)
        else:
            raise ValueError('BCELoss only adopts models with output type of logit or probability')
        hparams.update({
            'pos_weight{nme}'.format(nme=node_name()): node.BCELOSS.pos_weight,
            'loss-reduction{nme}'.format(nme=node_name()): node.BCELOSS.reduction
        })
        node.BCELOSS._visible(True)
    elif func_name == 'BinaryFocalLoss':
        if node.OUTPUT_TYPE == 'prob':
            loss_func = BinaryFocalLoss(**node.BINARYFOCALLOSS, **node.CUSTOM_ARGS)
        elif node.OUTPUT_TYPE == 'logit':
            loss_func = BinaryFocalWithLogitsLoss(**node.BINARYFOCALLOSS, **node.CUSTOM_ARGS)
        else:
            raise ValueError('BinaryFocalLoss only adopts models with output type of logit or probability')
        hparams.update({
            'focal-alpha{nme}'.format(nme=node_name()): node.BINARYFOCALLOSS.alpha,
            'focal-gamma{nme}'.format(nme=node_name()): node.BINARYFOCALLOSS.gamma,
            'loss-reduction{nme}'.format(nme=node_name()): node.BINARYFOCALLOSS.reduction
        })
        node.BINARYFOCALLOSS._visible(True)
    elif func_name == 'GeneralizedBinaryFocalLoss':
        if node.OUTPUT_TYPE == 'prob':
            loss_func = GeneralizedBinaryFocalLoss(**node.BINARYFOCALLOSS, **node.CUSTOM_ARGS)
        elif node.OUTPUT_TYPE == 'logit':
            loss_func = lambda input, target: GeneralizedBinaryFocalLoss(**node.BINARYFOCALLOSS, **node.CUSTOM_ARGS)(input.sigmoid(), target)
        else:
            raise ValueError('GeneralizedBinaryFocalLoss only adopts models with output type of logit or probability')
        hparams.update({
            'focal-alpha{nme}'.format(nme=node_name()): node.BINARYFOCALLOSS.alpha,
            'focal-gamma{nme}'.format(nme=node_name()): node.BINARYFOCALLOSS.gamma,
            'loss-reduction{nme}'.format(nme=node_name()): node.BINARYFOCALLOSS.reduction
        })
        node.BINARYFOCALLOSS._visible(True)
    elif func_name == 'FocalLoss':
        if node.OUTPUT_TYPE == 'prob':
            loss_func = FocalLoss(**node.FOCALLOSS, **node.CUSTOM_ARGS)
        elif node.OUTPUT_TYPE == 'logit':
            loss_func = lambda input, target: FocalLoss(**node.FOCALLOSS, **node.CUSTOM_ARGS)(F.softmax(input, dim=1), target)
        else:
            raise ValueError('FocalLoss only adopts models with output type of logit or probability')
        hparams.update({
            'focal-alpha{nme}'.format(nme=node_name()): node.FOCALLOSS.alpha,
            'focal-gamma{nme}'.format(nme=node_name()): node.FOCALLOSS.gamma,
            'balance-index{nme}'.format(nme=node_name()): node.FOCALLOSS.balance_index,
            'loss-reduction{nme}'.format(nme=node_name()): node.FOCALLOSS.reduction
        })
        node.FOCALLOSS._visible(True)
    elif func_name == 'GeneralizedFocalLoss':
        if node.OUTPUT_TYPE == 'prob':
            loss_func = GeneralizedFocalLoss(**node.FOCALLOSS, **node.CUSTOM_ARGS)
        elif node.OUTPUT_TYPE == 'logit':
            loss_func = lambda input, target: GeneralizedFocalLoss(**node.FOCALLOSS, **node.CUSTOM_ARGS)(F.softmax(input, dim=1), target)
        else:
            raise ValueError('FocalLoss only adopts models with output type of logit or probability')
        hparams.update({
            'focal-alpha{nme}'.format(nme=node_name()): node.FOCALLOSS.alpha,
            'focal-gamma{nme}'.format(nme=node_name()): node.FOCALLOSS.gamma,
            'balance-index{nme}'.format(nme=node_name()): node.FOCALLOSS.balance_index,
            'loss-reduction{nme}'.format(nme=node_name()): node.FOCALLOSS.reduction
        })
        node.FOCALLOSS._visible(True)
    elif func_name == 'DiceLoss':
        if node.OUTPUT_TYPE == 'prob':
            loss_func = DiceLoss(**node.DICELOSS, **node.CUSTOM_ARGS)
        elif node.OUTPUT_TYPE == 'logit':
            loss_func = lambda input, target: DiceLoss(**node.DICELOSS, **node.CUSTOM_ARGS)(
                input.sigmoid() if node.DICELOSS.task == 'binary' else F.softmax(input, dim=1),
                target
            )
        else:
            raise ValueError('DiceLoss only adopts models with output type of logit or probability')
        hparams.update({
            'dice-mode{nme}'.format(nme=node_name()): node.DICELOSS.mode,
            'dice-threshold{nme}'.format(nme=node_name()): node.DICELOSS.trheshold,
            'dice-channelreduction{nme}'.format(nme=node_name()): node.DICELOSS.channel_reduction,
            'loss-reduction{nme}'.format(nme=node_name()): node.DICELOSS.average,
        })
        node.DICELOSS._visible(True)
    elif func_name == 'GeneralizedDiceLoss':
        if node.OUTPUT_TYPE == 'prob':
            loss_func = GeneralizedDiceLoss(
                task=node.DICELOSS.task, average=node.DICELOSS.average, epsilon=node.DICELOSS.epsilon, **node.CUSTOM_ARGS
            )
        elif node.OUTPUT_TYPE == 'logit':
            loss_func = lambda input, target: GeneralizedDiceLoss(
                task=node.DICELOSS.task, average=node.DICELOSS.average, epsilon=node.DICELOSS.epsilon, **node.CUSTOM_ARGS
            )(input.sigmoid() if node.DICELOSS.task == 'binary' else F.softmax(input, dim=1), target)
        else:
            raise ValueError('GeneralizedDiceLoss only adopts models with output type of logit or probability')
        hparams.update({
            'loss-reduction{nme}'.format(nme=node_name()): node.DICELOSS.average,
        })
        node.DICELOSS.set_invisible_keys(['mode', 'channel_reduction', 'threshold'])
        node.DICELOSS._visible(True)
    elif func_name == 'CrossEntropyLoss':
        if node.OUTPUT_TYPE in ['logprob', 'real']:
            raise ValueError('CrossEntropyLoss only adopts models with output type of logit or probability')
        loss_func = CrossEntropyLoss(**node.CELOSS, **node.CUSTOM_ARGS)
        hparams.update({
            'loss-reduction{nme}'.format(nme=node_name()): node.CELOSS.reduction,
            'label-smoothine{nme}'.format(nme=node_name()): node.CELOSS.label_smoothing
        })
        node.CELOSS._visible(True)
    elif func_name == 'KLDivLoss':
        if node.OUTPUT_TYPE == 'real':
            raise ValueError('KLDivLoss do not adopts models with output type of real')
        elif node.OUTPUT_TYPE == 'logit':
            loss_func = lambda input, target: KLDivLoss(**node.KLDIVLOSS, **node.CUSTOM_ARGS)(input.sigmoid().log(), target)
        elif node.OUTPUT_TYPE == 'prob':
            loss_func = lambda input, target: KLDivLoss(**node.KLDIVLOSS, **node.CUSTOM_ARGS)(input.log(), target)
        else: # logprob
            loss_func = KLDivLoss(**node.KLDIVLOSS, **node.CUSTOM_ARGS)
        hparams.update({
            'loss-reduction{nme}'.format(nme=node_name()): node.KLDIVLOSS.reduction
        })
        node.KLDIVLOSS._visible(True)
    elif func_name == 'MSELoss':
        if node.OUTPUT_TYPE == 'logprob':
            raise ValueError('MSELoss do not adopts models with output type of logprob')
        elif node.OUTPUT_TYPE == 'logit':
            loss_func = lambda input, target: MSELoss(**node.MSELOSS, **node.CUSTOM_ARGS)(input.sigmoid(), target)
        else: # prob, real
            loss_func = MSELoss(**node.MSELOSS, **node.CUSTOM_ARGS)
        hparams.update({
            'loss-reduction{nme}'.format(nme=node_name()): node.MSELOSS.reduction
        })
        node.MSELOSS._visible(True)
    elif func_name == 'SmoothL1Loss':
        if node.OUTPUT_TYPE == 'logprob':
            raise ValueError('SmoothL1Loss do not adopts models with output type of logprob')
        elif node.OUTPUT_TYPE == 'logit':
            loss_func = lambda input, target: SmoothL1Loss(**node.SMOOTHL1LOSS, **node.CUSTOM_ARGS)(input.sigmoid(), target)
        else: # prob, real
            loss_func = SmoothL1Loss(**node.SMOOTHL1LOSS, **node.CUSTOM_ARGS)
        hparams.update({
            'loss-reduction{nme}'.format(nme=node_name()): node.SMOOTHL1LOSS.reduction,
            'smooth-beta{nme}'.format(nme=node_name()): node.SMOOTHL1LOSS.beta
        })
        node.SMOOTHL1LOSS._visible(True)
    elif func_name == 'HuberLoss':
        if node.OUTPUT_TYPE == 'logprob':
            raise ValueError('HuberLoss do not adopts models with output type of logprob')
        elif node.OUTPUT_TYPE == 'logit':
            loss_func = lambda input, target: HuberLoss(**node.HUBERLOSS, **node.CUSTOM_ARGS)(input.sigmoid(), target)
        else: # prob, real
            loss_func = HuberLoss(**node.HUBERLOSS, **node.CUSTOM_ARGS)
        hparams.update({
            'loss-reduction{nme}'.format(nme=node_name()): node.HUBERLOSS.reduction,
            'huber-delta{nme}'.format(nme=node_name()): node.HUBERLOSS.delta
        })
        node.HUBERLOSS._visible(True)
    elif func_name == 'SmoothL1NormalizedBCELoss':
        device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")  # Caution! This line is not of general purpose
        if node.OUTPUT_TYPE == 'prob':
            loss_func = SmoothL1NormalizedBCELoss(
                beta=node.SLN_BCELOSS.beta,
                label_smoothing=node.SLN_BCELOSS.label_smoothing,
                reduction=node.SLN_BCELOSS.reduction,
                pos_weight=torch.Tensor(node.SLN_BCELOSS.pos_weight).to(device),
                **node.CUSTOM_ARGS
            )
        elif node.OUTPUT_TYPE == 'logit':
            loss_func = SmoothL1NormalizedBCEWithLogitsLoss(
                beta=node.SLN_BCELOSS.beta,
                label_smoothing=node.SLN_BCELOSS.label_smoothing,
                reduction=node.SLN_BCELOSS.reduction,
                pos_weight=torch.Tensor(node.SLN_BCELOSS.pos_weight).to(device),
                **node.CUSTOM_ARGS
            )
        else:
            ValueError('SmoothL1NormalizedBCELoss only adopts models with output type of logit or probability')
        hparams.update({
            'pos_weight{nme}'.format(nme=node_name()): node.SLN_BCELOSS.pos_weight,
            'smooth-beta{nme}'.format(nme=node_name()): node.SLN_BCELOSS.beta,
            'label-smoothing{nme}'.format(nme=node_name()): node.SLN_BCELOSS.label_smoothing,
            'loss-reduction{nme}'.format(nme=node_name()): node.SLN_BCELOSS.reduction,
        })
        node.SLN_BCELOSS._visible(True)
    elif func_name == 'OrderedCrossEntropyLoss':
        if node.OUTPUT_TYPE in ['logprob', 'real']:
            raise ValueError('OrderedCrossEntropyLoss only adopts models with output type of logit and probability')
        elif node.OUTPUT_TYPE == 'prob':
            warnings.warn(
                'OrderedCrossEntropyLoss should use unnormalized logits as input.\n'
                + 'When probabilities are used as input, it may not throw an explicit error,\n'
                + 'but the loss calculation results may not be what the user expects.')
        loss_func = OrderedCrossEntropyLoss(
            num_classes=node.OCELOSS.num_classes,
            class_values=node.OCELOSS.class_values,
            reduction=node.CELOSS.reduction,
            label_smoothing=node.CELOSS.label_smoothing,
            **node.CUSTOM_ARGS
        )
        hparams.update({
            'num-classes{nme}'.format(nme=node_name()): node.OCELOSS.num_classes,
            'class_values{nme}'.format(nme=node_name()): node.OCELOSS.class_values,
            'loss-reduction{nme}'.format(nme=node_name()): node.CELOSS.reduction,
            'label-smoothine{nme}'.format(nme=node_name()): node.CELOSS.label_smoothing
        })
        node.OCELOSS._visible(True)
    elif func_name == 'PositionLoss':
        if node.OUTPUT_TYPE in ['logprob', 'real']:
            raise ValueError('PositionLoss only adopts models with output type of logit and probability')
        loss_func = PositionLoss(
            reduction=node.POSITIONLOSS.reduction,
            normalize=node.POSITIONLOSS.normalize,
            **node.CUSTOM_ARGS
        )
        hparams.update({
            'loss-reduction{nme}'.format(nme=node_name()): node.POSITIONLOSS.reduction,
            'normalize{nme}'.format(nme=node_name()): node.POSITIONLOSS.normalize,
        })
        node.POSITIONLOSS._visible(True)
    # ---------------Compose loss function getter --------------------------------
    elif func_name == 'Compose':
        loss_funcs = []
        loss_func_list = node.COMPOSE.LIST
        loss_func_weights = node.COMPOSE.WEIGHT
        hparams['loss-function{nme}'.format(nme=node_name())] = ' + '.join(
            ['{w}{abbr}'.format(
                w=loss_func_weights[i],
                abbr=node.ABBR[loss_func_list[i]]
            ) for i in range(len(loss_func_list))]
        )
        for _func_name in loss_func_list:
            node.defrost()
            node.CUSTOM_ARGS = node.CUSTOM_ARGS.get(_func_name, CN())
            node.freeze()
            _func, _hparam = get_lossfunc(node, _func_name, False)
            loss_funcs.append(_func)
            hparams.update(_hparam)
        loss_func = lambda pred, target: sum([
            loss_func_weights[i] * loss_funcs[i](pred, target) for i in range(len(loss_funcs))
        ])
        node.COMPOSE._visible(True)

    elif func_name == 'BCEWithLogitsLoss': # for backward compatibility
        if node.OUTPUT_TYPE in ['prob', 'logprob', 'real']:
            raise ValueError('BCEWithLogitsLoss only adopts models with output type of logit')
        device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda") # Caution! This line is not of general purpose
        loss_func = BCEWithLogitsLoss(reduction=node.BCELOSS.reduction, pos_weight=torch.Tensor(node.BCELOSS.pos_weight).to(device))
        hparams.update({
            'pos_weight{nme}'.format(nme=node_name()): node.BCELOSS.pos_weight,
            'loss-reduction{nme}'.format(nme=node_name()): node.BCELOSS.reduction
        })
        node.BCELOSS._visible(True)
    elif func_name == 'BinaryFocalWithLogitsLoss': # for backward compatibility
        if node.OUTPUT_TYPE in ['prob', 'logprob', 'real']:
            raise ValueError('BinaryFocalWithLogitsLoss only adopts models with output type of logit')
        loss_func = BinaryFocalWithLogitsLoss(**node.BINARYFOCALLOSS)
        hparams.update({
            'focal-alpha{nme}'.format(nme=node_name()): node.BINARYFOCALLOSS.alpha,
            'focal-gamma{nme}'.format(nme=node_name()): node.BINARYFOCALLOSS.gamma,
            'loss-reduction{nme}'.format(nme=node_name()): node.BINARYFOCALLOSS.reduction
        })
        node.BINARYFOCALLOSS._visible(True)
    elif func_name == 'FocalWithLogitsLoss': # for backward compatibility
        if node.OUTPUT_TYPE in ['prob', 'logprob', 'real']:
            raise ValueError('FocalWithLogitsLoss only adopts models with output type of logit')
        loss_func = lambda input, target: FocalLoss(**node.FOCALLOSS)(torch.softmax(input, dim=1), target)
        hparams.update({
            'focal-alpha{nme}'.format(nme=node_name()): node.FOCALLOSS.alpha,
            'focal-gamma{nme}'.format(nme=node_name()): node.FOCALLOSS.gamma,
            'balance-index{nme}'.format(nme=node_name()): node.FOCALLOSS.balance_index,
            'loss-reduction{nme}'.format(nme=node_name()): node.FOCALLOSS.reduction
        })
        node.FOCALLOSS._visible(True)
    elif func_name == 'DiceWithLogitsLoss': # for backward compatibility
        if node.OUTPUT_TYPE in ['logprob', 'real']:
            raise ValueError('DiceWithLogitsLoss only adopts models with output type of logit')
        loss_func = lambda input, target: DiceLoss(**node.DICELOSS)(
            input.sigmoid() if node.DICELOSS.task == 'binary' else F.softmax(input, dim=1),
            target
        )
        hparams.update({
            'dice-mode{nme}'.format(nme=node_name()): node.DICELOSS.mode,
            'dice-threshold{nme}'.format(nme=node_name()): node.DICELOSS.trheshold,
            'dice-channelreduction{nme}'.format(nme=node_name()): node.DICELOSS.channel_reduction,
            'loss-reduction{nme}'.format(nme=node_name()): node.DICELOSS.average,
        })
        node.DICELOSS._visible(True)
    elif func_name == 'SmoothL1NormalizedBCEWithLogitLoss': # for backward compatibility
        if node.OUTPUT_TYPE in ['prob', 'logprob', 'real']:
            raise ValueError('SmoothL1NormalizedBCEWithLogitLoss only adopts models with output type of logit')
        device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")  # Caution! This line is not of general purpose
        loss_func = SmoothL1NormalizedBCEWithLogitsLoss(
            beta=node.SLN_BCELOSS.beta,
            label_smoothing=node.SLN_BCELOSS.label_smoothing,
            reduction=node.SLN_BCELOSS.reduction,
            pos_weight=torch.Tensor(node.SLN_BCELOSS.pos_weight).to(device)
        )
        hparams.update({
            'pos_weight{nme}'.format(nme=node_name()): node.SLN_BCELOSS.pos_weight,
            'smooth-beta{nme}'.format(nme=node_name()): node.SLN_BCELOSS.beta,
            'label-smoothing{nme}'.format(nme=node_name()): node.SLN_BCELOSS.label_smoothing,
            'loss-reduction{nme}'.format(nme=node_name()): node.SLN_BCELOSS.reduction,
        })
        node.SLN_BCELOSS._visible(True)

    else:
        raise NotImplementedError

    return loss_func, hparams
