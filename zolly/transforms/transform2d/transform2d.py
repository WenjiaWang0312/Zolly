import torch


# Warp elements in image space to UV space.
def warp_feature(dp_out, feature_map, uv_res):
    """
    C: channel number of the input feature map;  H: height;  W: width

    :param dp_out: IUV image in shape (batch_size, 3, H, W)
    :param feature_map: Local feature map in shape (batch_size, C, H, W)
    :param uv_res: The resolution of the transferred feature map in UV space.

    :return: warped_feature: Feature map in UV space with shape (batch_size, C+3, uv_res, uv_res)
    The x, y cordinates in the image sapce and mask will be added as the last 3 channels
     of the warped feature, so the channel number of warped feature is C+3.
    """
    assert dp_out.shape[0] == feature_map.shape[0]
    assert dp_out.shape[2] == feature_map.shape[2]
    assert dp_out.shape[3] == feature_map.shape[3]

    dp_mask = dp_out[:, 0].unsqueeze(
        1)  # I channel, confidence of being foreground
    dp_uv = dp_out[:, 1:]  # UV channels, UV coordinates
    thre = 0.5  # The threshold of foreground and background.
    B, C, H, W = feature_map.shape
    device = feature_map.device

    # Get the sampling index of every pixel in batch_size dimension.
    index_batch = torch.arange(0, B, device=device,
                               dtype=torch.long)[:, None,
                                                 None].expand([-1, H, W])
    index_batch = index_batch.contiguous().view(-1).long()

    # Get the sampling index of every pixel in H and W dimension.
    tmp_x = torch.arange(0, W, device=device, dtype=torch.long)
    tmp_y = torch.arange(0, H, device=device, dtype=torch.long)

    y, x = torch.meshgrid(tmp_y, tmp_x)
    y = y.contiguous().view(-1).repeat([B])
    x = x.contiguous().view(-1).repeat([B])

    # Sample the confidence of every pixel,
    # and only preserve the pixels belong to foreground.
    conf = dp_mask[index_batch, 0, y, x].contiguous()
    valid = conf > thre
    index_batch = index_batch[valid]
    x = x[valid]
    y = y[valid]

    # Sample the uv coordinates of foreground pixels
    uv = dp_uv[index_batch, :, y, x].contiguous()
    num_pixel = uv.shape[0]
    # Get the corresponding location in UV space
    uv = uv * (uv_res - 1)
    uv_round = uv.round().long().clamp(min=0, max=uv_res - 1)

    # We first process the transferred feature in shape (batch_size * H * W, C+3),
    # so we need to get the location of each pixel in the two-dimension feature vector.
    index_uv = (uv_round[:, 1] * uv_res +
                uv_round[:, 0]).detach() + index_batch * uv_res * uv_res

    # Sample the feature of foreground pixels
    sampled_feature = feature_map[index_batch, :, y, x]
    # Scale x,y coordinates to [-1, 1] and
    # concatenated to the end of sampled feature as extra channels.
    y = (2 * y.float() / (H - 1)) - 1
    x = (2 * x.float() / (W - 1)) - 1
    sampled_feature = torch.cat([sampled_feature, x[:, None], y[:, None]],
                                dim=-1)

    # Multiple pixels in image space may be transferred to the same location in the UV space.
    # warped_w is used to record the number of the pixels transferred to every location.
    warped_w = sampled_feature.new_zeros([B * uv_res * uv_res, 1])
    warped_w.index_add_(0, index_uv, sampled_feature.new_ones([num_pixel, 1]))

    # Transfer the sampled feature to UV space.
    # Feature vectors transferred to the sample location will be accumulated.
    warped_feature = sampled_feature.new_zeros([B * uv_res * uv_res, C + 2])
    warped_feature.index_add_(0, index_uv, sampled_feature)

    # Normalize the accumulated feature with the pixel number.
    warped_feature = warped_feature / (warped_w + 1e-8)
    # Concatenate the mask channel at the end.
    warped_feature = torch.cat([warped_feature, (warped_w > 0).float()],
                               dim=-1)
    # Reshape the shape to (batch_size, C+3, uv_res, uv_res)
    warped_feature = warped_feature.reshape(B, uv_res, uv_res,
                                            C + 3).permute(0, 3, 1, 2)
    # warped_feature = warped_feature.reshape(B, uv_res, uv_res,
    #                                         C).permute(0, 3, 1, 2)
    return warped_feature[:, :C]