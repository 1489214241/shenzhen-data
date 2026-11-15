"""坐标转换：小区原始坐标是百度 BD-09，统一换成 WGS84。"""

import math


def gcj_offset(lon, lat):
    x, y = lon - 105, lat - 35
    a = -100 + 2 * x + 3 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    a += (20 * math.sin(6 * x * math.pi) + 20 * math.sin(2 * x * math.pi)) * 2 / 3
    a += (20 * math.sin(y * math.pi) + 40 * math.sin(y / 3 * math.pi)) * 2 / 3
    a += (160 * math.sin(y / 12 * math.pi) + 320 * math.sin(y * math.pi / 30)) * 2 / 3
    b = 300 + x + 2 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    b += (20 * math.sin(6 * x * math.pi) + 20 * math.sin(2 * x * math.pi)) * 2 / 3
    b += (20 * math.sin(x * math.pi) + 40 * math.sin(x / 3 * math.pi)) * 2 / 3
    b += (150 * math.sin(x / 12 * math.pi) + 300 * math.sin(x / 30 * math.pi)) * 2 / 3
    r = lat * math.pi / 180
    magic = 1 - 0.00669342162296594323 * math.sin(r) ** 2
    s = math.sqrt(magic)
    return b * 180 / (6378245 / s * math.cos(r) * math.pi), a * 180 / (
        (6378245 * (1 - 0.00669342162296594323)) / (magic * s) * math.pi
    )


def gcj_to_wgs(p):
    x, y = p
    a, b = x, y
    for _ in range(8):
        dx, dy = gcj_offset(a, b)
        a -= a + dx - x
        b -= b + dy - y
    return [a, b]


def bd_to_wgs(p):
    x, y = p[0] - 0.0065, p[1] - 0.006
    z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * math.pi * 3000 / 180)
    t = math.atan2(y, x) - 0.000003 * math.cos(x * math.pi * 3000 / 180)
    return gcj_to_wgs([z * math.cos(t), z * math.sin(t)])
