import torch
from torchvision.models import resnet50
import torchvision
import torch_pruning as pruning
import numpy as np

def prune_model(model):
    # model.cpu()
    # DG = pruning.DependencyGraph().build_dependency( model, torch.randn(1, 3, 224, 224) )
    # def prune_conv(conv, num_pruned):
    #     weight = conv.weight.detach().cpu().numpy()
    #     L1_norm = np.abs(weight)
    #     prune_index = np.argsort(L1_norm)[:num_pruned].tolist() # remove filters with small L1-Norm
    #     plan = DG.get_pruning_plan(conv, pruning.prune_batchnorm, prune_index)
    #     # print(plan)
    #     plan.exec()

    # blk_id = 0
    # for m in model.modules():
    #     #print(m)
    #     if isinstance(m, torchvision.models.resnet.Bottleneck):
    #         prune_conv(m.bn1, 10)
    #         prune_conv(m.bn2, 10)

    # return model  
    model.cpu()
    DG = pruning.DependencyGraph().build_dependency( model, torch.randn(1, 3, 224, 224) )
    def prune_conv(conv, pruned_prob):
        weight = conv.weight.detach().cpu().numpy()
        out_channels = weight.shape[0]
        L1_norm = np.abs(weight)
        # L1_norm = np.sum(np.abs(weight), axis=(1,2,3))
        num_pruned = int(out_channels * pruned_prob)
        prune_index = np.argsort(L1_norm)[:num_pruned].tolist() # remove filters with small L1-Norm
        plan = DG.get_pruning_plan(conv, pruning.prune_batchnorm, prune_index)
        plan.exec()
    
    block_prune_probs = [0.8]*16
    blk_id = 0
    for m in model.modules():
        if isinstance( m, torchvision.models.resnet.Bottleneck ):
            prune_conv( m.bn1, block_prune_probs[blk_id] )
            prune_conv( m.bn2, block_prune_probs[blk_id] )
            blk_id+=1
    return model  

model = resnet50(pretrained=True)

prune_model(model)
print(model)