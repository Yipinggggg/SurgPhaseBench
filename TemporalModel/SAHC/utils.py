# see https://github.com/xmed-lab/SAHC/blob/main/utils.py
import torch.nn.functional as F

def fusion(predicted_list, labels, args):
    all_out_list = []
    resize_out_list = []
    labels_list = []
    all_out = 0
    len_layer = len(predicted_list)
    weight_list = [1.0 / len_layer for i in range(0, len_layer)]
    # print(weight_list)
    num = 0
    for out, w in zip(predicted_list, weight_list):
        resize_out = F.interpolate(out, size=labels.size(0), mode='nearest')
        resize_out_list.append(resize_out)
        # align_corners=True
        # print(out.size())
        resize_label = F.interpolate(labels.float().unsqueeze(0).unsqueeze(0), size=out.size(2), mode='linear',
                                     align_corners=False)
        if out.size(2) == labels.size(0):
            resize_label = labels
            labels_list.append(resize_label.squeeze().long())
        else:
            # resize_label = max_pool(labels_list[-1].float().unsqueeze(0).unsqueeze(0))
            resize_label = F.interpolate(labels.float().unsqueeze(0).unsqueeze(0), size=out.size(2), mode='nearest')
            # resize_label2 = F.interpolate(resize_label,size=labels.size(0),mode='nearest')
            # ,align_corners=True
            # print(resize_label.size(), resize_label2.size())
            # print((resize_label2 == labels).sum()/labels.size(0))
            # with open(path_p+'{}.txt'.format(num),"w") as f:
            #     for labl1, lab2 in zip(resize_label2.squeeze(), labels.squeeze()):
            #         f.writelines(str(labl1)+'\t'+str(lab2)+'\n')
            # num+=1
            labels_list.append(resize_label.squeeze().long())
            # labels_list.append(labels.squeeze().long())
        # print(resize_label.size(), out.size())
        # labels_list.append(labels.squeeze().long())
        # assert resize_out.size(2) == resize_label.size(0)
        # assert resize_label.size(2) == out.size(2)
        # print(out.size())
        # print(resize_label.size())
        # print(resize_out.size())
        # all_out_list.append(out)
        # all_out_list.append(resize_out)

        all_out_list.append(out)
        # resize_out=out
        # all_out = all_out + w*resize_out

    # sss
    return all_out, all_out_list, labels_list