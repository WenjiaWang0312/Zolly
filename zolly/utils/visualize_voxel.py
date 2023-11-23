import matplotlib.pyplot as plt
import numpy as np


def vis_voxel(voxel,
              im_size,
              dpi=100,
              face_color=None,
              edge_color=[1, 1, 1, 1],
              no_axis=False,
              return_im=False):
    # and plot everything

    fig = plt.figure(figsize=(im_size / dpi, im_size / dpi), dpi=dpi)
    ax = fig.add_subplot(projection='3d')
    if face_color is None:
        face_color = np.array([1, 1, 1, 1])
    elif isinstance(face_color, (tuple, list)):
        face_color = np.array(face_color)
    ax.voxels(voxel, facecolors=face_color, edgecolor=np.array(edge_color))
    if no_axis:
        ax.axis('off')
    fig.canvas.draw()
    if return_im:
        img_data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] +
                                    (4, ))
        return img_data
    else:
        plt.show()


def vis_heatmap(heatmap,
                im_size,
                no_axis=False,
                channel='xyz',
                signs=[1, 1, 1],
                clip_value=0.2,
                return_im=False):  # heatmap: (J, dim, dim, dim)

    heatmap = heatmap.transpose(0, 3, 2, 1)
    voxels = np.sum(heatmap, axis=0)  # (dim, dim, dim)
    channel = [channel.index(c) for c in 'xyz']
    voxels = voxels.transpose(channel)  # (dim, dim, dim) -> (dim, dim, dim)
    for axis, sign in enumerate(signs):
        if sign == -1:
            voxels = np.flip(voxels,
                             axis=axis)  # (dim, dim, dim) -> (dim, dim, dim)
    # normalize
    voxels = voxels / np.max(voxels)
    colors = plt.cm.viridis(voxels)
    #cvtcolor
    colors = colors[..., [2, 1, 0, 3]]
    voxels = (voxels > clip_value) * voxels
    return vis_voxel(voxels,
                     im_size,
                     dpi=100,
                     face_color=colors,
                     edge_color=[0., 0., 0., 0],
                     no_axis=no_axis,
                     return_im=return_im)
