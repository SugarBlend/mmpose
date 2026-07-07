import cv2
import numpy as np
import torch
from mmpose.visualization.simcc_vis import SimCCVisualizer


class GroupedSimCCVisualizer(SimCCVisualizer):
    def split_simcc_xy(self, heatmap: torch.Tensor, indices: list[int] = None) -> tuple[list[dict[str, torch.Tensor]], list[int]]:
        size = heatmap.size()
        if indices is None:
            indices = list(range(size[0] if size[0] <= 20 else 20))
        maps = []
        for idx in indices:
            xy_dict = {}
            single_heatmap = heatmap[idx]
            xy_dict['x'], xy_dict['y'] = self.merge_maps(single_heatmap)
            maps.append(xy_dict)
        return maps, indices

    def draw_instance_heatmap(
        self, heatmap: torch.Tensor, overlaid_image: np.ndarray, n: int = 20, mix: bool = True, weight: float = 0.5,
        indices: list[int] = None
    ) -> np.ndarray:
        return self.draw_instance_xy_heatmap(heatmap, overlaid_image, n, mix, weight, indices)

    def draw_instance_xy_heatmap(
        self, heatmap: torch.Tensor, overlaid_image: np.ndarray, n: int = 20, mix: bool = True, weight: float = 0.5,
        indices: list[int] = None
    ) -> np.ndarray:
        heatmap2d = heatmap.data.max(0, keepdim=True)[0]
        xy_heatmap, indices = self.split_simcc_xy(heatmap, indices=indices)
        K = len(indices)
        blank_size = tuple(heatmap.size()[1:])
        maps = {'x': [], 'y': []}
        for i in xy_heatmap:
            x, y = self.draw_1d_heatmaps(i['x']), self.draw_1d_heatmaps(i['y'])
            maps['x'].append(x)
            maps['y'].append(y)
        white = self.creat_blank(blank_size, K)
        map2d = self.draw_2d_heatmaps(heatmap2d)
        if mix:
            map2d = cv2.addWeighted(overlaid_image, 1 - weight, map2d, weight, 0)
        self.image_cover(white, map2d, int(blank_size[1] * 0.1), int(blank_size[0] * 0.1))
        white = self.add_1d_heatmaps(maps, white, blank_size, K, indices=indices)
        return white

    def add_1d_heatmaps(
        self,
        maps: dict[str, list[np.ndarray]],
        background: np.ndarray,
        map2d_size: tuple[int, int],
        K: int,
        interval: int = 10,
        indices: list[int] = None
    ) -> np.ndarray:
        if indices is None:
            indices = list(range(K))

        y_startpoint, x_startpoint = ([int(1.1 * map2d_size[1]), int(0.1 * map2d_size[0])],
                                      [int(0.1 * map2d_size[1]), int(1.1 * map2d_size[0])])
        x_startpoint[1] += interval * 2
        y_startpoint[0] += interval * 2
        add = interval + 10
        for i, idx in enumerate(indices):
            self.image_cover(background, maps['x'][i], x_startpoint[0], x_startpoint[1])
            cv2.putText(background, str(idx),(x_startpoint[0] - 30, x_startpoint[1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            self.image_cover(background, maps['y'][i], y_startpoint[0], y_startpoint[1])
            cv2.putText(background, str(idx),(y_startpoint[0], y_startpoint[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            x_startpoint[1] += add
            y_startpoint[0] += add
        return background[:x_startpoint[1] + y_startpoint[1] + 1, :y_startpoint[0] + x_startpoint[0] + 1]


class GroupedHeatmapVisualizer(SimCCVisualizer):
    def draw_instance_heatmap(
        self, heatmap: torch.Tensor, overlaid_image: np.ndarray, indices: list[int] = None, mix: bool = True,
        weight: float = 0.5, cols: int = 6, cell_scale: float = 1.0, interpolation: int  = cv2.INTER_CUBIC
    ) -> np.ndarray:
        K_total = heatmap.shape[0]
        H_native, W_native = heatmap.shape[1], heatmap.shape[2]

        # Calculate the heatmap at the native resolution, then upscale it to the resolution of the input image -
        # this way, the sharpness of the original crop is preserved, not the heatmap grid.
        combined = heatmap[: K_total].amax(dim=0, keepdim=True)  # (1, H, W)
        map2d = self.draw_2d_heatmaps(combined)  # color map, native H, W
        map2d = cv2.resize(map2d, overlaid_image.shape[-2:-4:-1], interpolation=interpolation)
        if mix:
            map2d = cv2.addWeighted(overlaid_image, 1 - weight, map2d, weight, 0)

        if indices:
            cell_h, cell_w = int(H_native * cell_scale), int(W_native * cell_scale)
            pad, label_h = 4, 16
            n = len(indices)
            cols = min(cols, n)
            rows = int(np.ceil(n / cols))

            grid_h = rows * (cell_h + label_h + pad) + pad
            grid_w = cols * (cell_w + pad) + pad
            grid = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)

            for i, idx in enumerate(indices):
                r, c = divmod(i, cols)
                single = heatmap[idx:idx + 1]  # (1, H, W)
                cell_img = self.draw_2d_heatmaps(single)
                cell_img = cv2.resize(cell_img, (cell_w, cell_h))
                y0 = pad + r * (cell_h + label_h + pad) + label_h
                x0 = pad + c * (cell_w + pad)
                self.image_cover(grid, cell_img, x0, y0)
                cv2.putText(grid, str(idx), (x0, y0 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            top_h, top_w = map2d.shape[:2]
            canvas_w = max(top_w, grid_w)
            canvas = np.full((top_h + grid_h + pad, canvas_w, 3), 255, dtype=np.uint8)
            canvas[:top_h, :top_w] = map2d
            canvas[top_h + pad:top_h + pad + grid_h, :grid_w] = grid
            return canvas
        else:
            return map2d
