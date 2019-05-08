import torch
import torch.nn.functional as F


def style_swap(content_feature, style_feature, kernel_size, stride=1):
    # content_feature and style_feature should have shape as (1, C, H, W)
    # kernel_size here is equivalent to extracted patch size

    # extract patches from style_feature with shape (1, C, H, W)
    kh, kw = kernel_size, kernel_size
    sh, sw = stride, stride

    patches = style_feature.unfold(2, kh, sh).unfold(3, kw, sw)

    patches = patches.permute(0, 2, 3, 1, 4, 5)
    patches = patches.reshape(-1, *patches.shape[-3:]) # (patch_numbers, C, kh, kw)

    # calculate Frobenius norm and normalize the patches at each filter
    norm = torch.norm(patches.reshape(patches.shape[0], -1), dim=1).reshape(-1, 1, 1, 1)

    noramalized_patches = patches / norm

    conv_out = F.conv2d(content_feature, noramalized_patches)

    # calculate the argmax at each spatial location, which means at each (kh, kw),
    # there should exist a filter which provides the biggest value of the output
    one_hots = torch.zeros_like(conv_out)
    one_hots.scatter_(1, conv_out.argmax(dim=1, keepdim=True), 1)

    # deconv/transpose conv
    deconv_out = F.conv_transpose2d(one_hots, patches)

    # calculate the overlap from deconv/transpose conv
    overlap = F.conv_transpose2d(one_hots, torch.ones_like(patches))

    # average the deconv result
    res = deconv_out / overlap
    return res

#
# c = torch.arange(27).reshape(1, 3, 3, 3).float()
# s = torch.arange(27).reshape(1, 3, 3, 3).float()
# #
# style_swap(c, s, 2, 1).shape
