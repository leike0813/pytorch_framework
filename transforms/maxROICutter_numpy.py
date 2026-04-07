"""
maxROICutter 的纯 NumPy 实现，替代 SymPy 符号计算以提高性能。

算法逻辑与原版相同，仅将几何计算改用 NumPy 数值实现。
"""
from typing import List, Tuple, Optional
import numpy as np


__all__ = [
    'ImageWithIrregularROI'
]


class PointWrapper:
    """
    点包装类，兼容 SymPy Point 的属性访问方式。

    支持：
    - .x, .y 属性访问
    - [0], [1] 索引访问
    - NumPy 数组操作
    - 基本算术运算（通过 NumPy）
    """
    __slots__ = ('_coords')

    def __init__(self, coords: np.ndarray):
        self._coords = coords.astype(np.float64)

    @property
    def x(self) -> float:
        return self._coords[0]

    @property
    def y(self) -> float:
        return self._coords[1]

    def __getitem__(self, idx):
        return self._coords[idx]

    def __array__(self, dtype=None):
        """支持 NumPy 数组操作。"""
        if dtype is not None:
            return self._coords.astype(dtype)
        return self._coords.copy()

    def __sub__(self, other):
        """支持减法运算。"""
        if isinstance(other, PointWrapper):
            return self._coords - other._coords
        return self._coords - np.asarray(other)

    def __add__(self, other):
        """支持加法运算。"""
        if isinstance(other, PointWrapper):
            return self._coords + other._coords
        return self._coords + np.asarray(other)

    def __repr__(self):
        return f"Point({self.x}, {self.y})"


def _segment_horizontal_ray_intersection(
    segment_p1: np.ndarray,
    segment_p2: np.ndarray,
    y: float
) -> Optional[float]:
    """
    计算线段与水平射线(y固定)的交点x坐标。

    Args:
        segment_p1: 线段起点 [x, y]
        segment_p2: 线段终点 [x, y]
        y: 水平射线的y坐标

    Returns:
        交点的x坐标，如果无交点则返回 None
    """
    dy = segment_p2[1] - segment_p1[1]
    if dy == 0:
        return None  # 线段水平，与水平射线平行或重合

    t = (y - segment_p1[1]) / dy
    if t < 0 or t > 1:
        return None  # 交点不在线段上

    x = segment_p1[0] + t * (segment_p2[0] - segment_p1[0])
    return x


def _segment_vertical_ray_intersection(
    segment_p1: np.ndarray,
    segment_p2: np.ndarray,
    x: float
) -> Optional[float]:
    """
    计算线段与垂直射线(x固定)的交点y坐标。

    Args:
        segment_p1: 线段起点 [x, y]
        segment_p2: 线段终点 [x, y]
        x: 垂直射线的x坐标

    Returns:
        交点的y坐标，如果无交点则返回 None
    """
    dx = segment_p2[0] - segment_p1[0]
    if dx == 0:
        return None  # 线段垂直，与垂直射线平行或重合

    t = (x - segment_p1[0]) / dx
    if t < 0 or t > 1:
        return None  # 交点不在线段上

    y = segment_p1[1] + t * (segment_p2[1] - segment_p1[1])
    return y


def _point_in_convex_quadrangle(point: np.ndarray, quad_points: np.ndarray) -> bool:
    """
    判断点是否在凸四边形内部（半平面法）。

    对于凸四边形，点在内部 = 点在所有四条边的同一侧（内侧）。

    Args:
        point: 待检测点 [x, y]
        quad_points: 四边形顶点数组 shape (4, 2)，顺序为左上、右上、右下、左下

    Returns:
        True 如果点在四边形内部（含边界）
    """
    # 四条边的向量表示：edge_i = p_{i+1} - p_i
    edges = np.array([
        quad_points[1] - quad_points[0],  # top: p1 -> p2
        quad_points[2] - quad_points[1],  # right: p2 -> p3
        quad_points[3] - quad_points[2],  # bottom: p3 -> p4
        quad_points[0] - quad_points[3],  # left: p4 -> p1
    ])

    # 从每条边起点到检测点的向量
    to_point = np.array([
        point - quad_points[0],
        point - quad_points[1],
        point - quad_points[2],
        point - quad_points[3],
    ])

    # 计算 2D 叉积 (edge x to_point)
    # 正值表示点在边向量的左侧，负值表示右侧
    cross = edges[:, 0] * to_point[:, 1] - edges[:, 1] * to_point[:, 0]

    # 对于顺时针排列的四边形，内部点应该在所有边的右侧（叉积 <= 0）
    # 对于逆时针排列的四边形，内部点应该在所有边的左侧（叉积 >= 0）
    all_negative_or_zero = np.all(cross <= 1e-10)
    all_positive_or_zero = np.all(cross >= -1e-10)

    return all_negative_or_zero or all_positive_or_zero


def _compute_slope(p1: np.ndarray, p2: np.ndarray) -> float:
    """计算两点连线的斜率，垂直线返回 np.inf。"""
    dx = p2[0] - p1[0]
    if dx == 0:
        return np.inf
    return (p2[1] - p1[1]) / dx


class QuadrangleMixin:
    """
    四边形 Mixin 类，提供顶点访问和几何操作方法。

    不存储顶点数据，依赖子类提供 `points` 属性（shape (2, 4)）。
    """

    @property
    def list_points(self) -> List[List[int]]:
        """返回顶点坐标列表，每列是一个顶点 [x, y]。"""
        return [self.points[:, i].astype(np.int64).tolist() for i in range(4)]

    @property
    def p1(self) -> PointWrapper:  # topleft
        return PointWrapper(self.points[:, 0])

    @property
    def p2(self) -> PointWrapper:  # topright
        return PointWrapper(self.points[:, 1])

    @property
    def p3(self) -> PointWrapper:  # bottomright
        return PointWrapper(self.points[:, 2])

    @property
    def p4(self) -> PointWrapper:  # bottomleft
        return PointWrapper(self.points[:, 3])

    topleft = p1
    topright = p2
    bottomright = p3
    bottomleft = p4

    @property
    def edge1(self) -> Tuple[np.ndarray, np.ndarray]:  # top: p1 -> p2
        return (self.points[:, 0].copy(), self.points[:, 1].copy())

    @property
    def edge2(self) -> Tuple[np.ndarray, np.ndarray]:  # right: p2 -> p3
        return (self.points[:, 1].copy(), self.points[:, 2].copy())

    @property
    def edge3(self) -> Tuple[np.ndarray, np.ndarray]:  # bottom: p3 -> p4
        return (self.points[:, 2].copy(), self.points[:, 3].copy())

    @property
    def edge4(self) -> Tuple[np.ndarray, np.ndarray]:  # left: p4 -> p1
        return (self.points[:, 3].copy(), self.points[:, 0].copy())

    edge_top = edge1
    edge_right = edge2
    edge_bottom = edge3
    edge_left = edge4

    @property
    def edge4_slope(self) -> float:
        """edge4 (左边，p4->p1) 的斜率。"""
        return _compute_slope(self.p4, self.p1)

    @property
    def edge1_slope(self) -> float:
        """edge1 (顶边，p1->p2) 的斜率。"""
        return _compute_slope(self.p1, self.p2)

    def is_topleft_superfluous(self) -> bool:
        """
        判断左上角是否是"多余的"顶点。

        原算法逻辑：
        - 如果 edge1 斜率 < 0: edge4 斜率 < 0 或 edge4 斜率 == oo 时为多余
        - 如果 edge1 斜率 == 0: edge4 斜率 < 0 时为多余
        - 否则不多余
        """
        edge1_slope = self.edge1_slope
        edge4_slope = self.edge4_slope

        if edge1_slope < 0:
            return edge4_slope < 0 or edge4_slope == np.inf
        elif edge1_slope == 0:
            return edge4_slope < 0
        return False

    def cut_topleft_superfluous(self):
        """
        处理左上角多余顶点，通过发射射线"修剪"四边形。

        原算法逻辑：
        - 如果 edge1.slope < 0: 从 p1 发射水平向右射线，与 edge2 求交得到新 p2
        - 如果 edge4.slope < 0: 从 p1 发射垂直向下射线，与 edge3 求交得到新 p4
        """
        if not self.is_topleft_superfluous():
            return

        # edge1.slope < 0 时，修剪 p2
        edge1_slope = self.edge1_slope
        if edge1_slope < 0:
            # 从 p1 (左上) 发射水平向右射线
            # 射线与 edge2 (右边) 的交点作为新 p2
            p1 = self.p1
            edge2_p1, edge2_p2 = self.edge2  # p2 -> p3
            y = p1[1]
            # 水平射线 y = p1.y，与 edge2 求交
            x_intersect = _segment_horizontal_ray_intersection(edge2_p1, edge2_p2, y)
            if x_intersect is not None:
                new_p2 = np.array([x_intersect, y])
                # 使用 set_new_point (如果存在) 或直接修改
                if hasattr(self, 'set_new_point'):
                    try:
                        self.set_new_point(new_p2, 1)  # p2 是索引 1
                    except ValueError:
                        pass  # 点超出边界时忽略

        # edge4.slope < 0 时，修剪 p4
        edge4_slope = self.edge4_slope
        if edge4_slope < 0:
            # 从 p1 (edge4.p2 = p1) 发射垂直向下射线
            # 射线与 edge3 (底边) 的交点作为新 p4
            p1 = self.p1
            edge3_p1, edge3_p2 = self.edge3  # p3 -> p4
            x = p1[0]
            # 垂直射线 x = p1.x，与 edge3 求交
            y_intersect = _segment_vertical_ray_intersection(edge3_p1, edge3_p2, x)
            if y_intersect is not None:
                new_p4 = np.array([x, y_intersect])
                if hasattr(self, 'set_new_point'):
                    try:
                        self.set_new_point(new_p4, 3)  # p4 是索引 3
                    except ValueError:
                        pass  # 点超出边界时忽略

    def search_orthogonal_rectangle(self, max_grid: int = 100) -> Tuple[float, Optional[np.ndarray]]:
        """
        在四边形内搜索最大的内接轴对齐矩形。

        算法：沿着 edge4 (左边，p4->p1) 遍历一系列点，对于每个点：
        1. 向右发射水平射线，与目标边(edge1或edge3)求交
        2. 向上或向下发射垂直射线，与目标边求交
        3. 从这两个交点出发再发射射线，确定矩形的第四个角
        4. 检查第四个角是否在四边形内部
        5. 计算面积，保留最大面积的矩形

        Args:
            max_grid: 遍历的最大步数

        Returns:
            (max_area, max_rect_points):
            - max_area: 最大矩形面积，-1 表示失败
            - max_rect_points: 最大矩形的四个顶点坐标 shape (4, 2)，失败时为 None
        """
        if self.is_topleft_superfluous():
            print('Cannot apply current algorithm to Quadrangle with superfluous vertex.')
            return -1, None

        # edge4: p4 -> p1 (左边，从下到上)
        p4 = np.asarray(self.p4)
        p1 = np.asarray(self.p1)
        edge4_vec = p1 - p4  # 从 p4 指向 p1 的向量

        # 确定遍历步数
        edge4_x_pixels = abs(p1[0] - p4[0])
        edge4_y_pixels = abs(p1[1] - p4[1])
        edge4_pixels = int(min(max_grid, max(edge4_x_pixels, edge4_y_pixels)))

        if edge4_pixels <= 1:
            return -1, None

        # 根据斜率确定射线方向和目标边
        edge4_slope = self.edge4_slope
        if edge4_slope < 0:
            ray1_dir = 1  # 水平向右
            ray2_dir = 1  # 垂直向下
            ray1_tar = self.edge1  # top edge: p1 -> p2
            ray2_tar = self.edge3  # bottom edge: p3 -> p4
        elif edge4_slope > 0:
            ray1_dir = 1  # 水平向右
            ray2_dir = -1  # 垂直向上
            ray1_tar = self.edge3  # bottom edge: p3 -> p4
            ray2_tar = self.edge1  # top edge: p1 -> p2
        else:
            # edge4 垂直或水平的情况
            ray1_dir = 1
            ray2_dir = 1
            ray1_tar = self.edge1
            ray2_tar = self.edge3

        # 四边形顶点用于包含检测
        quad_vertices = np.array([
            np.asarray(self.p1),
            np.asarray(self.p2),
            np.asarray(self.p3),
            np.asarray(self.p4)
        ])

        max_area = -1.0
        max_rect = None

        for i in range(edge4_pixels - 1):
            # 计算当前遍历点（在 edge4 上，从上往下遍历）
            # SymPy: edge4.arbitrary_point() 返回 p4 + t*(p1-p4)
            # subs(var, 1 - (i+1)/edge4_pixels) 即 t = 1 - (i+1)/edge4_pixels
            # 当 i=0, t ≈ 0.99, point ≈ p4 + 0.99*(p1-p4) ≈ p1 (接近上端)
            t = 1.0 - (i + 1) / edge4_pixels
            point = p4 + t * edge4_vec  # 在 edge4 上的点
            point_int = np.round(point).astype(np.int64)

            # 发射水平射线 (向右，y = point[1])，与目标边求交
            tar1_p1, tar1_p2 = ray1_tar
            x_intersect = _segment_horizontal_ray_intersection(tar1_p1, tar1_p2, point_int[1])
            if x_intersect is None:
                continue

            # 发射垂直射线 (向上或向下，x = point[0])，与目标边求交
            tar2_p1, tar2_p2 = ray2_tar
            y_intersect = _segment_vertical_ray_intersection(tar2_p1, tar2_p2, point_int[0])
            if y_intersect is None:
                continue

            # 计算矩形顶点
            intersect1 = np.array([x_intersect, point_int[1]])  # 水平射线交点
            intersect2 = np.array([point_int[0], y_intersect])  # 垂直射线交点
            intersect3 = np.array([x_intersect, y_intersect])  # 第四角
            intersect3_int = np.round(intersect3).astype(np.int64)

            # 检查第四个角是否在四边形内部
            if not _point_in_convex_quadrangle(intersect3_int, quad_vertices):
                continue

            # 计算矩形面积
            width = abs(x_intersect - point_int[0])
            height = abs(y_intersect - point_int[1])
            area = width * height

            if area > max_area:
                max_area = area
                # 构建矩形顶点（顺序需要与四边形顶点顺序一致：左上、右上、右下、左下）
                if ray2_dir == 1:  # 垂直向下，point 是左上角
                    max_rect = np.array([
                        point_int,          # 左上 (起点)
                        intersect1,         # 右上 (水平交点)
                        intersect3_int,     # 右下 (第四角)
                        intersect2          # 左下 (垂直交点)
                    ])
                else:  # 垂直向上，point 是左下角
                    max_rect = np.array([
                        intersect2,         # 左上 (垂直交点)
                        intersect3_int,     # 右上 (第四角)
                        intersect1,         # 右下 (水平交点)
                        point_int           # 左下 (起点)
                    ])

        return max_area, max_rect


class ImageMixin:
    """
    图像边界 Mixin 类。

    不存储宽高数据，依赖子类提供 `width` 和 `height` 属性。
    """

    @property
    def corner_topleft(self) -> np.ndarray:
        return np.array([0.0, 0.0])

    @property
    def corner_topright(self) -> np.ndarray:
        return np.array([self.width - 1, 0.0])

    @property
    def corner_bottomright(self) -> np.ndarray:
        return np.array([self.width - 1, self.height - 1])

    @property
    def corner_bottomleft(self) -> np.ndarray:
        return np.array([0.0, self.height - 1])

    c1 = corner_topleft
    c2 = corner_topright
    c3 = corner_bottomright
    c4 = corner_bottomleft

    @property
    def bounding_polygon(self) -> np.ndarray:
        return np.array([self.c1, self.c2, self.c3, self.c4])

    @property
    def size(self) -> Tuple[int, int]:
        return (self.width, self.height)

    def in_image(self, point: np.ndarray) -> bool:
        return (0 <= point[0] < self.width) and (0 <= point[1] < self.height)


class ImageView(QuadrangleMixin, ImageMixin):
    """
    图像视图类，支持不同视角（旋转90度的倍数）查看四边形。

    通过 permutation 和 transformation 矩阵实现坐标转换。
    """

    permute_matrices = [
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        np.array([[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]),
        np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]),
        np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]),
    ]
    trans_matrices = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 1], [-1, 0]]),
        np.array([[-1, 0], [0, -1]]),
        np.array([[0, -1], [1, 0]]),
    ]
    bias_index = [[-1, -1], [-1, 0], [0, 1], [1, -1]]
    inverse_trans_matrices = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, -1], [1, 0]]),
        np.array([[-1, 0], [0, -1]]),
        np.array([[0, 1], [-1, 0]]),
    ]
    inverse_bias_index = [[-1, -1], [0, -1], [0, 1], [-1, 1]]

    def __init__(self, image: 'ImageWithIrregularROI', perspective: int):
        self._image = image
        self._perspective = perspective

    @property
    def image(self):
        return self._image

    @property
    def perspective(self):
        return self._perspective

    @property
    def width(self) -> int:
        return self._image.width if self._perspective == 0 or self._perspective == 2 else self._image.height

    @property
    def height(self) -> int:
        return self._image.height if self._perspective == 0 or self._perspective == 2 else self._image.width

    @property
    def points(self) -> np.ndarray:
        """变换后的顶点坐标 shape (2, 4)。"""
        trans_mat = self.trans_matrices[self._perspective]
        bias = np.array([
            self._image.size[i] - 1 if i >= 0 else 0
            for i in self.bias_index[self._perspective]
        ]).reshape(2, 1)
        perm_mat = self.permute_matrices[self._perspective]
        transformed = np.matmul(trans_mat, self._image.points) + bias
        permuted = np.matmul(transformed, perm_mat)
        return permuted

    def set_new_point(self, point: np.ndarray, point_idx: int):
        """设置新顶点（反向变换回原始坐标系）。"""
        if not self.in_image(point):
            raise ValueError('Point beyond image boundary.')

        inv_trans_mat = self.inverse_trans_matrices[self._perspective]
        inv_bias = np.array([
            self._image.size[i] - 1 if i >= 0 else 0
            for i in self.inverse_bias_index[self._perspective]
        ])

        # 反向变换
        new_point = np.array([
            np.dot(inv_trans_mat[i], point) + inv_bias[i]
            for i in range(2)
        ])

        # 找到对应的原始顶点索引
        perm_mat = self.permute_matrices[self._perspective]
        orig_idx = np.argmax(perm_mat.T[point_idx])
        self._image._points[:, orig_idx] = new_point

    def set_new_ROI_poly(self, ROI_poly: np.ndarray):
        """设置新的 ROI 四边形。"""
        for i in range(4):
            self.set_new_point(ROI_poly[i], i)


class ImageWithIrregularROI(QuadrangleMixin, ImageMixin):
    """
    带有不规则 ROI 的图像类。

    支持旋转、翻转等变换，并能计算最大内接矩形。
    """

    def __init__(self, width: int, height: int, ROI: List[List[int]]):
        self.width = width
        self.height = height
        self._points = np.array(ROI, dtype=np.float64).T  # shape (2, 4)
        self.views = [ImageView(self, i) for i in range(4)]

    @property
    def points(self) -> np.ndarray:
        return self._points

    @points.setter
    def points(self, value: np.ndarray):
        self._points = value.astype(np.float64)

    def view(self, perspective: int) -> ImageView:
        return self.views[perspective]

    def rotate(self, angle: int):
        """旋转 ROI 四边形。"""
        rotate_center = np.array([self.width / 2.0, self.height / 2.0])
        cos_a = np.cos(angle * np.pi / 180.0)
        sin_a = np.sin(angle * np.pi / 180.0)

        # 旋转边界四边形的四个角
        corners = self.bounding_polygon
        rotated_corners = []
        for corner in corners:
            rel = corner - rotate_center
            new_rel = np.array([rel[0] * cos_a - rel[1] * sin_a,
                                rel[0] * sin_a + rel[1] * cos_a])
            rotated_corners.append(new_rel + rotate_center)

        rotated_corners = np.array(rotated_corners)
        min_x = np.floor(rotated_corners[:, 0].min())
        max_x = np.ceil(rotated_corners[:, 0].max())
        min_y = np.floor(rotated_corners[:, 1].min())
        max_y = np.ceil(rotated_corners[:, 1].max())

        new_width = int(max_x - min_x + 3)
        new_height = int(max_y - min_y + 3)

        # 旋转 ROI 顶点
        roi_corners = self._points.T  # shape (4, 2)
        rotated_roi = []
        for corner in roi_corners:
            rel = corner - rotate_center
            new_rel = np.array([rel[0] * cos_a - rel[1] * sin_a,
                                rel[0] * sin_a + rel[1] * cos_a])
            new_corner = new_rel + rotate_center - np.array([min_x + 1, min_y + 1])
            rotated_roi.append(new_corner)

        rotated_roi = np.array(rotated_roi)

        # 按y排序，调整顺序为左上、右上、右下、左下
        new_points = rotated_roi.T  # shape (2, 4)
        sort_idx = np.argsort(new_points[1])
        if new_points[0, sort_idx[0]] > new_points[0, sort_idx[1]]:
            sort_idx[0], sort_idx[1] = sort_idx[1], sort_idx[0]
        if new_points[0, sort_idx[2]] < new_points[0, sort_idx[3]]:
            sort_idx[2], sort_idx[3] = sort_idx[3], sort_idx[2]

        self._points = new_points[:, sort_idx]
        self.width = new_width
        self.height = new_height
        self.views = [ImageView(self, i) for i in range(4)]

    def hflip(self):
        """水平翻转。"""
        self._points = np.array([
            [self.width - 1 - self.p2[0], self.p2[1]],
            [self.width - 1 - self.p1[0], self.p1[1]],
            [self.width - 1 - self.p4[0], self.p4[1]],
            [self.width - 1 - self.p3[0], self.p3[1]],
        ], dtype=np.float64).T
        self.views = [ImageView(self, i) for i in range(4)]

    def vflip(self):
        """垂直翻转。"""
        self._points = np.array([
            [self.p4[0], self.height - 1 - self.p4[1]],
            [self.p3[0], self.height - 1 - self.p3[1]],
            [self.p2[0], self.height - 1 - self.p2[1]],
            [self.p1[0], self.height - 1 - self.p1[1]],
        ], dtype=np.float64).T
        self.views = [ImageView(self, i) for i in range(4)]

    def is_superfluous_ROI(self) -> bool:
        """检查 ROI 是否有多余顶点。"""
        return any(view.is_topleft_superfluous() for view in self.views)

    def is_orthogonal_rectangle(self) -> bool:
        """检查 ROI 是否是正交矩形（四个角都是 90 度且顶边水平）。"""
        # 检查顶边是否水平 (p1.y == p2.y)
        if abs(self.p1[1] - self.p2[1]) > 1e-6:
            return False

        # 检查四个角是否都是 90 度
        # 通过检查相邻边的向量是否垂直（点积为零）
        edges = [
            self.p2 - self.p1,  # top
            self.p3 - self.p2,  # right
            self.p4 - self.p3,  # bottom
            self.p1 - self.p4,  # left
        ]
        for i in range(4):
            dot = np.dot(edges[i], edges[(i + 1) % 4])
            if abs(dot) > 1e-6:
                return False

        return True

    def cut_superfluous_ROI(self):
        """循环处理多余顶点，直到四边形变成正交矩形。"""
        max_iterations = 100  # 防止无限循环
        for iteration in range(max_iterations):
            if self.is_orthogonal_rectangle():
                break
            for view in self.views:
                view.cut_topleft_superfluous()
            # 重新创建 views，因为 points 可能已改变
            self.views = [ImageView(self, i) for i in range(4)]

    def cut_non_superfluous_ROI(self):
        """寻找最大内接矩形并设置为新 ROI。"""
        max_area = -1.0
        max_poly = None
        max_poly_perspective = -1

        for i, view in enumerate(self.views):
            area, poly = view.search_orthogonal_rectangle()
            if area > max_area:
                max_area = area
                max_poly = poly
                max_poly_perspective = i

        if max_poly is not None:
            self.view(max_poly_perspective).set_new_ROI_poly(max_poly)

    def cut_max_ROI(self):
        """执行最大 ROI 切割。"""
        # 如果已经是正交矩形，无需处理
        if self.is_orthogonal_rectangle():
            return

        # 如果有多余顶点，先处理
        if self.is_superfluous_ROI():
            self.cut_superfluous_ROI()
            return

        # 搜索最大内接矩形
        self.cut_non_superfluous_ROI()


if __name__ == "__main__":
    import time

    roi = [[100, 50], [400, 80], [380, 350], [120, 320]]
    cutter = ImageWithIrregularROI(500, 400, roi)

    print("原始 ROI:", cutter.list_points)

    start = time.time()
    cutter.cut_max_ROI()
    elapsed = time.time() - start

    print("切割后 ROI:", cutter.list_points)
    print(f"耗时: {elapsed * 1000:.2f} ms")