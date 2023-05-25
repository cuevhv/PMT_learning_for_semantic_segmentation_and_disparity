import torch
import torch.nn.functional as F

def process(left, right, model):
    if left.shape[2] % 16 != 0:
            times = left.shape[2]//16       
            top_pad = (times+1)*16 -left.shape[2]
    else:
        top_pad = 0

    if left.shape[3] % 16 != 0:
        times = left.shape[3]//16                       
        right_pad = (times+1)*16-left.shape[3]
    else:
        right_pad = 0    

    left = F.pad(left,(0,right_pad, top_pad,0))#.unsqueeze(0)
    right = F.pad(right,(0,right_pad, top_pad,0))#.unsqueeze(0)


    outputs = model(left, right)
    outputs=torch.squeeze(outputs)
    # outputs = outputs.data.cpu().numpy()
    if top_pad !=0 or right_pad != 0:
        outputs = outputs[top_pad:,:-right_pad]
    else:
        outputs = outputs
    
    # plt.imshow(outputs.cpu().numpy())
    # plt.show()
    return outputs.unsqueeze(0).unsqueeze(0)