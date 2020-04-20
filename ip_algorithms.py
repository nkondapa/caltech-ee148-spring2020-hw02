import numpy as np


def targeted_filtering(I, points, kernels):

    if len(I.shape) != 3:
        I = np.expand_dims(I, axis=2)
    (n_rows, n_cols, n_channels) = np.shape(I)

    heatmaps = []
    for T in kernels:
        heatmap = np.zeros(shape=(n_rows, n_cols))

        if len(T.shape) == 2 and n_channels != 1:
            (Trows, Tcols) = np.shape(T)
            T = np.expand_dims(T, axis=2)
            repT = np.tile(T, reps=3)
        elif len(T.shape) == 3 and n_channels == 1:
            T = np.mean(T, axis=2)
            (Trows, Tcols) = np.shape(T)
            repT = T
        else:
            (Trows, Tcols, Tchannels) = np.shape(T)
            repT = T

        tr_Trows = int(Trows / 2)
        br_Trows = Trows - tr_Trows
        lc_Tcols = int(Tcols / 2)
        rc_Tcols = Tcols - lc_Tcols

        padded_patch = np.zeros(shape=(T.shape[0], T.shape[1], n_channels))
        surround_points = []
        for p in points:
            i, j = p

            tr = max(0, i - tr_Trows)
            br = min(n_rows, i + br_Trows)
            lc = max(0, j - lc_Tcols)
            rc = min(n_cols, j + rc_Tcols)

            offsetr = abs(i - tr_Trows - tr) + abs(i + br_Trows - br)
            offsetc = abs(j - lc_Tcols - lc) + abs(j + rc_Tcols - rc)
            patch = I[tr:br, lc:rc, :]
            surround_x = list(range(tr, br))
            surround_y = list(range(lc, rc))
            surround_points.extend(list(zip(surround_x, surround_y)))
            # print(i, j)

            if patch.shape != repT.shape:
                # print(i, j, offsetr, offsetc)
                # print(patch.shape)
                padded_patch[offsetr:, offsetc:, :] = patch
                patch = padded_patch

            patch_norm = (1 / np.linalg.norm(patch.flatten()))
            T_norm = (1 / np.linalg.norm(repT.flatten()))
            heatmap[i, j] = np.dot(patch.flatten(), repT.flatten()) * patch_norm * T_norm

        for p in surround_points:
            i, j = p

            tr = max(0, i - tr_Trows)
            br = min(n_rows, i + br_Trows)
            lc = max(0, j - lc_Tcols)
            rc = min(n_cols, j + rc_Tcols)

            offsetr = abs(i - tr_Trows - tr) + abs(i + br_Trows - br)
            offsetc = abs(j - lc_Tcols - lc) + abs(j + rc_Tcols - rc)
            patch = I[tr:br, lc:rc, :]
            # print(i, j)

            if patch.shape != repT.shape:
                # print(i, j, offsetr, offsetc)
                # print(patch.shape)
                padded_patch[offsetr:, offsetc:, :] = patch
                patch = padded_patch

            patch_norm = (1 / np.linalg.norm(patch.flatten()))
            T_norm = (1 / np.linalg.norm(repT.flatten()))
            heatmap[i, j] = np.dot(patch.flatten(), repT.flatten()) * patch_norm * T_norm

        heatmaps.append(heatmap)

    return heatmaps
