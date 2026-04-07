"""
测试 maxROICutter 的 SymPy 版本与 NumPy 版本的一致性和性能对比。
"""
import sys
import time
import numpy as np
from typing import List, Tuple

# 添加项目根目录到 Python 路径
sys.path.insert(0, '/mnt/D/Workspace/Code/Python/CustomUtils/pytorch_framework')

import importlib.util

# 直接加载模块文件，避免通过 __init__.py 触发 torch 依赖
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

sympy_module = load_module('maxROICutter_sympy', '/mnt/D/Workspace/Code/Python/CustomUtils/pytorch_framework/transforms/maxROICutter.py')
numpy_module = load_module('maxROICutter_numpy_mod', '/mnt/D/Workspace/Code/Python/CustomUtils/pytorch_framework/transforms/maxROICutter_numpy.py')

SympyCutter = sympy_module.ImageWithIrregularROI
NumpyCutter = numpy_module.ImageWithIrregularROI


def generate_test_cases() -> List[Tuple[int, int, List[List[int]], int]]:
    """
    生成测试用例。

    Returns:
        List of (width, height, ROI, rotation_angle)
    """
    test_cases = []

    # 1. 基本矩形 ROI
    test_cases.append((500, 400, [[100, 50], [400, 80], [380, 350], [120, 320]], 0))

    # 2. 透视变换后的 ROI（模拟 RandomPerspective 的输出）
    test_cases.append((640, 480,
        [[50, 30], [600, 20], [620, 460], [40, 450]], 0))

    # 3. 较大旋转角度
    test_cases.append((800, 600,
        [[100, 100], [700, 100], [700, 500], [100, 500]], 45))

    # 4. 小图像
    test_cases.append((100, 100,
        [[10, 10], [90, 10], [90, 90], [10, 90]], 30))

    # 5. 大图像
    test_cases.append((1920, 1080,
        [[200, 100], [1720, 150], [1700, 950], [250, 900]], 0))

    # 6. 不规则四边形
    test_cases.append((400, 300,
        [[50, 200], [350, 50], [380, 250], [30, 280]], 0))

    # 7. 旋转后的极端情况
    test_cases.append((500, 500,
        [[100, 100], [400, 100], [400, 400], [100, 400]], 90))

    return test_cases


def compare_results(
    sympy_result: List[List[int]],
    numpy_result: List[List[int]],
    tolerance: int = 2
) -> bool:
    """
    比较两个结果是否在允许误差范围内一致。

    Args:
        sympy_result: SymPy 版本的输出
        numpy_result: NumPy 版本的输出
        tolerance: 允许的像素误差

    Returns:
        True 如果结果一致
    """
    sympy_arr = np.array(sympy_result)
    numpy_arr = np.array(numpy_result)

    diff = np.abs(sympy_arr - numpy_arr)
    max_diff = diff.max()

    if max_diff > tolerance:
        print(f"  结果差异: max_diff={max_diff}, tolerance={tolerance}")
        print(f"  SymPy 结果: {sympy_result}")
        print(f"  NumPy 结果: {numpy_result}")
        return False

    return True


def run_single_test(
    width: int,
    height: int,
    roi: List[List[int]],
    rotation: int,
    max_grid: int = 100
) -> Tuple[bool, float, float]:
    """
    运行单个测试用例。

    Returns:
        (passed, sympy_time_ms, numpy_time_ms)
    """
    # SymPy 版本
    sympy_cutter = SympyCutter(width, height, roi)
    if rotation != 0:
        sympy_cutter.rotate(-rotation)

    start = time.time()
    sympy_cutter.cut_max_ROI()
    sympy_time = (time.time() - start) * 1000

    sympy_result = sympy_cutter.list_points

    # NumPy 版本
    numpy_cutter = NumpyCutter(width, height, roi)
    if rotation != 0:
        numpy_cutter.rotate(-rotation)

    start = time.time()
    numpy_cutter.cut_max_ROI()
    numpy_time = (time.time() - start) * 1000

    numpy_result = numpy_cutter.list_points

    # 比较结果
    passed = compare_results(sympy_result, numpy_result)

    return passed, sympy_time, numpy_time


def test_consistency():
    """测试结果一致性。"""
    print("=" * 60)
    print("结果一致性测试")
    print("=" * 60)

    test_cases = generate_test_cases()
    passed_count = 0

    for i, (width, height, roi, rotation) in enumerate(test_cases):
        print(f"\n测试 #{i+1}: size={width}x{height}, rotation={rotation}")
        passed, sympy_time, numpy_time = run_single_test(width, height, roi, rotation)

        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  结果: {status}")
        print(f"  SymPy 耗时: {sympy_time:.2f} ms")
        print(f"  NumPy 耗时: {numpy_time:.2f} ms")
        print(f"  性能提升: {sympy_time / numpy_time:.1f}x")

        if passed:
            passed_count += 1

    print(f"\n总计: {passed_count}/{len(test_cases)} 测试通过")
    return passed_count == len(test_cases)


def test_performance():
    """性能对比测试。"""
    print("\n" + "=" * 60)
    print("性能对比测试 (多次运行取平均)")
    print("=" * 60)

    # 使用一个典型大小的测试用例
    width, height = 640, 480
    roi = [[50, 30], [600, 20], [620, 460], [40, 450]]

    n_runs = 10

    # SymPy 版本
    sympy_times = []
    for _ in range(n_runs):
        sympy_cutter = SympyCutter(width, height, roi)
        start = time.time()
        sympy_cutter.cut_max_ROI()
        sympy_times.append((time.time() - start) * 1000)

    sympy_avg = np.mean(sympy_times)
    sympy_std = np.std(sympy_times)

    # NumPy 版本
    numpy_times = []
    for _ in range(n_runs):
        numpy_cutter = NumpyCutter(width, height, roi)
        start = time.time()
        numpy_cutter.cut_max_ROI()
        numpy_times.append((time.time() - start) * 1000)

    numpy_avg = np.mean(numpy_times)
    numpy_std = np.std(numpy_times)

    print(f"\n测试配置: {width}x{height}, n_runs={n_runs}")
    print(f"SymPy: {sympy_avg:.2f} ± {sympy_std:.2f} ms")
    print(f"NumPy: {numpy_avg:.2f} ± {numpy_std:.2f} ms")
    print(f"性能提升: {sympy_avg / numpy_avg:.1f}x")


def test_rotation_sequence():
    """测试完整的变换序列（模拟 RandomTransformAndCutoff 的使用场景）。"""
    print("\n" + "=" * 60)
    print("完整变换序列测试")
    print("=" * 60)

    width, height = 512, 512
    # 模拟透视变换后的 ROI
    roi = [[50, 30], [480, 20], [500, 490], [40, 480]]
    rotation = 30

    print(f"\n初始配置: {width}x{height}")
    print(f"ROI: {roi}")
    print(f"旋转角度: {rotation}")

    # SymPy 版本
    sympy_cutter = SympyCutter(width, height, roi)
    print(f"\nSymPy - 旋转前: size={sympy_cutter.width}x{sympy_cutter.height}")

    start = time.time()
    sympy_cutter.rotate(-rotation)
    rotate_time_sympy = (time.time() - start) * 1000
    print(f"SymPy - 旋转后: size={sympy_cutter.width}x{sympy_cutter.height}")

    start = time.time()
    sympy_cutter.cut_max_ROI()
    cut_time_sympy = (time.time() - start) * 1000

    sympy_result = sympy_cutter.list_points

    # NumPy 版本
    numpy_cutter = NumpyCutter(width, height, roi)
    print(f"\nNumPy - 旋转前: size={numpy_cutter.width}x{numpy_cutter.height}")

    start = time.time()
    numpy_cutter.rotate(-rotation)
    rotate_time_numpy = (time.time() - start) * 1000
    print(f"NumPy - 旋转后: size={numpy_cutter.width}x{numpy_cutter.height}")

    start = time.time()
    numpy_cutter.cut_max_ROI()
    cut_time_numpy = (time.time() - start) * 1000

    numpy_result = numpy_cutter.list_points

    # 比较
    passed = compare_results(sympy_result, numpy_result)

    print(f"\n旋转耗时:")
    print(f"  SymPy: {rotate_time_sympy:.2f} ms")
    print(f"  NumPy: {rotate_time_numpy:.2f} ms")

    print(f"\n切割耗时:")
    print(f"  SymPy: {cut_time_sympy:.2f} ms")
    print(f"  NumPy: {cut_time_numpy:.2f} ms")
    print(f"  性能提升: {cut_time_sympy / cut_time_numpy:.1f}x")

    print(f"\n结果: {'✓ 通过' if passed else '✗ 失败'}")
    print(f"SymPy 最终 ROI: {sympy_result}")
    print(f"NumPy 最终 ROI: {numpy_result}")

    return passed


def test_edge_cases():
    """测试边界情况。"""
    print("\n" + "=" * 60)
    print("边界情况测试")
    print("=" * 60)

    cases = [
        # 小图像
        ("小图像 50x50", 50, 50, [[5, 5], [45, 5], [45, 45], [5, 45]], 0),
        # 几乎全图 ROI
        ("几乎全图 ROI", 200, 200, [[10, 10], [190, 10], [190, 190], [10, 190]], 0),
        # 狭长 ROI
        ("狭长 ROI", 400, 200, [[50, 50], [350, 50], [350, 150], [50, 150]], 0),
    ]

    passed_count = 0
    for name, width, height, roi, rotation in cases:
        print(f"\n{name}:")
        passed, sympy_time, numpy_time = run_single_test(width, height, roi, rotation)
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  结果: {status}")
        print(f"  性能提升: {sympy_time / numpy_time:.1f}x")
        if passed:
            passed_count += 1

    print(f"\n总计: {passed_count}/{len(cases)} 测试通过")
    return passed_count == len(cases)


def main():
    """运行所有测试。"""
    print("maxROICutter 性能优化测试")
    print("=" * 60)

    all_passed = True

    # 1. 结果一致性测试
    if not test_consistency():
        all_passed = False

    # 2. 性能对比测试
    test_performance()

    # 3. 完整变换序列测试
    if not test_rotation_sequence():
        all_passed = False

    # 4. 边界情况测试
    if not test_edge_cases():
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过 ✓")
    else:
        print("部分测试失败 ✗")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    main()