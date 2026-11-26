"""Mask-guided patch extraction and volume reassembly."""

import numpy as np


def block_ind(mask, sz_block=64, sz_pad=0):
    spatial = np.asarray(mask)
    if spatial.ndim == 4 and spatial.shape[-1] == 1:
        spatial = spatial[..., 0]
    if spatial.ndim != 3:
        raise ValueError("Expected a 3D spatial mask with an optional singleton channel")
    spans = [np.flatnonzero(spatial.any(axis=axes)) for axes in ((1, 2), (0, 2), (0, 1))]
    if not all(span.size for span in spans):
        raise ValueError("The mask is empty")
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = [(span[0], span[-1]) for span in spans]
    ind_brain = [xmin, xmax, ymin, ymax, zmin, zmax]

    xlen = xmax - xmin + 1
    ylen = ymax - ymin + 1
    zlen = zmax - zmin + 1

    nx = int(np.ceil(xlen / sz_block)) + sz_pad
    ny = int(np.ceil(ylen / sz_block)) + sz_pad
    nz = int(np.ceil(zlen / sz_block)) + sz_pad

    xstart = xmin
    ystart = ymin
    zstart = zmin

    xend = xmax - sz_block + 1
    yend = ymax - sz_block + 1
    zend = zmax - sz_block + 1

    xind_block = np.round(np.linspace(xstart, xend, nx))
    yind_block = np.round(np.linspace(ystart, yend, ny))
    zind_block = np.round(np.linspace(zstart, zend, nz))

    ind_block = np.zeros([xind_block.shape[0] * yind_block.shape[0] * zind_block.shape[0], 6])
    count = 0
    for ii in np.arange(0, xind_block.shape[0]):
        for jj in np.arange(0, yind_block.shape[0]):
            for kk in np.arange(0, zind_block.shape[0]):
                ind_block[count, :] = np.array([xind_block[ii], xind_block[ii] + sz_block - 1, yind_block[jj],
                                                yind_block[jj] + sz_block - 1, zind_block[kk], zind_block[kk] + sz_block - 1])
                count = count + 1

    ind_block = ind_block.astype(int)

    return ind_block, ind_brain


def extract_block(data, inds, *, dtype=np.float64):
    xsz_block = inds[0, 1] - inds[0, 0] + 1
    ysz_block = inds[0, 3] - inds[0, 2] + 1
    zsz_block = inds[0, 5] - inds[0, 4] + 1
    ch_block = data.shape[-1]

    blocks = np.empty((inds.shape[0], xsz_block, ysz_block, zsz_block, ch_block), dtype=dtype)
    for ii in np.arange(inds.shape[0]):
        inds_this = inds[ii, :]
        blocks[ii, :, :, :, :] = data[inds_this[0]:inds_this[1] + 1, inds_this[2]:inds_this[3] + 1,
                                      inds_this[4]:inds_this[5] + 1, :]

    return blocks


def block2brain(blocks, inds, mask, sz_crop=0):
    vol_brain = np.zeros([mask.shape[0], mask.shape[1], mask.shape[2], blocks.shape[-1]])
    vol_count = np.zeros([mask.shape[0], mask.shape[1], mask.shape[2], blocks.shape[-1]])

    indmax = np.max(inds, axis=0)
    indmin = np.min(inds, axis=0)

    for tt in np.arange(inds.shape[0]):
        inds_this = inds[tt, :].copy()
        inds0_this = np.array([0, blocks.shape[1], 0, blocks.shape[2], 0, blocks.shape[3]])

        if sz_crop > 0:
            for mm in np.arange(0, 6, 2):
                if inds_this[mm] == indmin[mm]:
                    continue
                else:
                    inds_this[mm] = inds_this[mm] + sz_crop
                    inds0_this[mm] = inds0_this[mm] + sz_crop

            for mm in np.arange(1, 6, 2):
                if inds_this[mm] == indmax[mm]:
                    continue
                else:
                    inds_this[mm] = inds_this[mm] - sz_crop
                    inds0_this[mm] = inds0_this[mm] - sz_crop

        vol_brain[inds_this[0]:inds_this[1] + 1, inds_this[2]:inds_this[3] + 1, inds_this[4]:inds_this[5] + 1, :] = \
            vol_brain[inds_this[0]:inds_this[1] + 1, inds_this[2]:inds_this[3] + 1, inds_this[4]:inds_this[5] + 1, :] + \
            blocks[tt, inds0_this[0]:inds0_this[1], inds0_this[2]:inds0_this[3], inds0_this[4]:inds0_this[5], :]

        vol_count[inds_this[0]:inds_this[1] + 1, inds_this[2]:inds_this[3] + 1, inds_this[4]:inds_this[5] + 1, :] = \
            vol_count[inds_this[0]:inds_this[1] + 1, inds_this[2]:inds_this[3] + 1, inds_this[4]:inds_this[5] + 1, :] + 1.

    vol_count[vol_count < 0.5] = 1.
    vol_brain = vol_brain / vol_count

    vol_brain = vol_brain * mask
    vol_count = vol_count * mask

    return vol_brain, vol_count


def block_indices(mask, block_size=64, overlap=1):
    """Return patch bounds, adding `overlap` positions per spatial axis."""
    inds, _ = block_ind(mask, sz_block=block_size, sz_pad=overlap)
    if np.any(inds[:, ::2] < 0) or np.any(inds[:, 1::2] >= np.array(mask.shape[:3])):
        raise ValueError(
            f"The mask-derived {block_size}-voxel patches do not fit inside the input grid. "
            "Use the preprocessed HCP grid with enough space around the brain mask."
        )
    return inds
