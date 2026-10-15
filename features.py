"""
Ядро анализа: детекция ключевых точек, сопоставление дескрипторов,
поиск общих точек, замер устойчивости метода.
Используется и скриптом визуализации keypoints.py, и генератором отчёта report.py.
"""
import os
import glob

import cv2
import numpy as np

IMAGES = sorted(glob.glob(os.path.join(os.path.dirname(__file__),
                                       "images", "*.jpg")))
TOP_N = 30
MIN_RATIO = 0.75
MAX_RATIO = 0.90


def make_detector():
    return cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02,
                           edgeThreshold=8, sigma=1.6)


def load():
    """Читает изображения, возвращает (цветные, серые, имена)."""
    images = [cv2.imread(p) for p in IMAGES]
    gray = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in images]
    names = [os.path.basename(p) for p in IMAGES]
    return images, gray, names


def detect_all(gray, detector=None):
    """Детекция на всех изображениях. Возвращает (kps, dess)."""
    detector = detector or make_detector()
    kps, dess = [], []
    for g in gray:
        k, d = detector.detectAndCompute(g, None)
        kps.append(k)
        dess.append(d)
    return kps, dess


def ratio_matches(d1, d2, ratio):
    """Ratio-тест Лоу: список (queryIdx, trainIdx, distance)."""
    if d1 is None or d2 is None:
        return []
    raw = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    out = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            out.append((m.queryIdx, m.trainIdx, m.distance))
    return out


def ratio_sets(d1, d2, ratio):
    return {q for q, t, d in ratio_matches(d1, d2, ratio)}


def _fwd_map(d1, d2, ratio):
    """queryIdx -> trainIdx при прохождении Ratio-теста (один лучший)."""
    out = {}
    for q, t, d in ratio_matches(d1, d2, ratio):
        out[q] = t
    return out


def mutual_matches(d1, d2, ratio):
    """Двустороннее (mutual) сопоставление: пара (q->t) признаётся только если
    обратно (t->q) тоже получается лучшей парой. Фильтрует неоднозначные пары."""
    fwd = _fwd_map(d1, d2, ratio)
    bwd = _fwd_map(d2, d1, ratio)
    return [(q, t) for q, t in fwd.items() if bwd.get(t) == q]


def verified_pairs(kps1, kps2, d1, d2, ratio, ransac_reproj=5.0):
    """Полностью отфильтрованная пара соответствий:
       1) Ratio-тест Лоу;
       2) взаимная (mutual) согласованность;
       3) геометрическая верификация RANSAC-гомографией (инлайеры).
    Возвращает список (queryIdx, trainIdx)."""
    pairs = mutual_matches(d1, d2, ratio)
    if len(pairs) >= 4:
        src = np.array([kps1[q].pt for q, t in pairs], dtype=np.float32).reshape(-1, 1, 2)
        dst = np.array([kps2[t].pt for q, t in pairs], dtype=np.float32).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_reproj)
        if mask is not None:
            pairs = [p for p, keep in zip(pairs, mask.ravel()) if keep]
    return pairs


def pair_matrix(dess, ratio=0.75):
    """Матрица числа взаимных соответствий между всеми парами изображений."""
    n = len(dess)
    M = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            c = len(ratio_matches(dess[i], dess[j], ratio))
            M[i, j] = M[j, i] = c
    return M


def pair_verified_matrix(kps, dess, ratio, ransac_reproj=5.0, mutual=True):
    """Матрица ОТФИЛЬТРОВАННЫХ соответствий между парами изображений
    (взаимная проверка + RANSAC). """
    n = len(dess)
    M = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if mutual:
                c = len(verified_pairs(kps[i], kps[j], dess[i], dess[j],
                                       ratio, ransac_reproj))
            else:
                c = len(mutual_matches(dess[i], dess[j], ratio))
            M[i, j] = M[j, i] = c
    return M


def find_common(kps, dess):
    """Выбирает эталон и порог Ratio так, чтобы максимизировать число общих
    точек. Возвращает словарь с результатами."""
    ratios = [r / 100.0 for r in range(int(MIN_RATIO * 100),
                                       int(MAX_RATIO * 100) + 1, 5)]
    best = (0, -1, -1, None)
    for ref in range(len(dess)):
        for ratio in ratios:
            sets = [ratio_sets(dess[ref], dess[i], ratio)
                    for i in range(len(dess)) if i != ref]
            common = set.intersection(*sets) if sets else set()
            if len(common) > best[0]:
                best = (len(common), ref, ratio, sets)

    n_common, ref, ratio, _ = best
    others_idx = [i for i in range(len(dess)) if i != ref]

    # Множества и надёжность для выбранного эталона/ratio.
    sets = [ratio_sets(dess[ref], dess[i], ratio) for i in others_idx]
    common = set.intersection(*sets) if sets else set()

    reli = {}
    for qi in common:
        total = 0.0
        for k, i in enumerate(others_idx):
            m = ratio_matches(dess[ref], dess[i], ratio)
            total += {q: d for q, t, d in m}.get(qi, float("inf"))
        reli[qi] = total

    ordered = sorted(reli, key=reli.get)[:TOP_N]
    numbering = {qi: num for num, qi in enumerate(ordered, start=1)}
    return {
        "ref": ref, "ratio": ratio, "n_common": n_common,
        "common": ordered, "numbering": numbering,
        "others_idx": others_idx, "sets": sets,
    }


def pair_matrix(dess, ratio=0.75):
    """Матрица числа взаимных соответствий между всеми парами изображений."""
    n = len(dess)
    M = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            c = len(ratio_matches(dess[i], dess[j], ratio))
            M[i, j] = M[j, i] = c
    return M


def assign_zones(coords):
    """Разбивает нормированные координаты (x,y in [0,1]) на анатомические
    зоны тела ракообразного. Возвращает (зона->число точек)."""
    zones = {}
    for (x, y) in coords:
        if y < 0.35:
            zone = "голова/рострум" if x < 0.5 else "спинной край карапакса"
        elif y > 0.65:
            zone = "вентральные придатки/конечности"
        else:
            zone = "центральная область карапакса"
        zones[zone] = zones.get(zone, 0) + 1
    return zones


# --------------------------- устойчивость метода ----------------------------
def perturb_rot(img, angle):
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h))


def perturb_scale(img, factor):
    return cv2.resize(img, None, fx=factor, fy=factor,
                      interpolation=cv2.INTER_LINEAR)


def perturb_blur(img, sigma):
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigma)


def perturb_brightness(img, factor):
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def robustness_curve(ref_gray, detector, perturb_fn, levels, ratio=0.75):
    """Число надёжных соответствий между оригиналом и perturbed-копией."""
    kp0, d0 = detector.detectAndCompute(ref_gray, None)
    counts = []
    for lvl in levels:
        copy = perturb_fn(ref_gray, lvl)
        kp1, d1 = detector.detectAndCompute(copy, None)
        if d1 is None:
            counts.append(0)
        else:
            counts.append(len(ratio_matches(d0, d1, ratio)))
    return counts