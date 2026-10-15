"""
ПОИСК ОБЩИХ КЛЮЧЕВЫХ ТОЧЕК НА НЕСКОЛЬКИХ ИЗОБРАЖЕНИЯХ ОРГАНИЗМОВ.

Детекция -> сопоставление (Ratio-тест Лоу) -> общие точки -> визуализация.
Подробнее в DESCRIPTORS.md и отчёте report.pdf (report.py).
"""
import os

import cv2
import matplotlib.pyplot as plt

import features as F


def main():
    images, gray, names = F.load()
    detector = F.make_detector()
    kps, dess = F.detect_all(gray, detector)

    print(f"Изображений: {len(images)}")
    print("\n1) Ключевых точек, найденных первоначально на каждом изображении:")
    for n, (name, k) in enumerate(zip(names, kps), start=1):
        print(f"   [{n}] {name}: {len(k)}")

    res = F.find_common(kps, dess)
    ref, ratio = res["ref"], res["ratio"]

    print(f"\n2) Соответствий между изображениями:")
    print(f"   a) после Ratio-теста (Ratio={ratio:.2f}):")
    M = F.pair_matrix(dess, ratio=ratio)
    print("        " + "".join(f"{j+1:6d}" for j in range(len(images))))
    for i in range(len(images)):
        print(f"   img {i+1}:" + "".join(f"{M[i, j]:6d}" for j in range(len(images))))
    print(f"   б) после двусторонней проверки и RANSAC-верификации:")
    Mv = F.pair_verified_matrix(kps, dess, ratio=ratio)
    print("        " + "".join(f"{j+1:6d}" for j in range(len(images))))
    for i in range(len(images)):
        print(f"   img {i+1}:" + "".join(f"{Mv[i, j]:6d}" for j in range(len(images))))

    print(f"\n3) Общих точек (во всех изображениях): {res['n_common']}")

    ordered = res["common"]
    numbering = res["numbering"]
    if ordered:
        import numpy as np
        pts = [kps[ref][qi].pt for qi in ordered]
        arr = np.array(pts, dtype=float)
        x0, y0 = arr.min(0)
        x1, y1 = arr.max(0)
        coords = [((p[0]-x0)/(x1-x0+1e-6), (p[1]-y0)/(y1-y0+1e-6)) for p in arr]
        zones = F.assign_zones(coords)
        print("\n4) Распределение устойчивых общих точек по зонам тела:")
        for zone, cnt in sorted(zones.items(), key=lambda kv: -kv[1]):
            print(f"   {zone}: {cnt}")
    else:
        print("\n4) Устойчивых общих точек нет — анализ зон тела не проводится.")

    colors = plt.cm.tab20.colors
    ncols = 3
    nrows = (len(images) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 8 * nrows))
    axes = axes.ravel()
    for img_idx, ax in enumerate(axes):
        if img_idx >= len(images):
            ax.axis("off")
            continue
        ax.imshow(cv2.cvtColor(images[img_idx], cv2.COLOR_BGR2RGB))
        ax.set_title(f"[{img_idx+1}] {names[img_idx]}", fontsize=11)
        if img_idx == ref:
            points = {qi: kps[ref][qi].pt for qi in numbering}
        else:
            rm = F.ratio_matches(dess[ref], dess[img_idx], ratio)
            train_of = {q: t for q, t, d in rm}
            points = {qi: kps[img_idx][train_of[qi]].pt for qi in numbering
                      if qi in train_of}
        for qi, (x, y) in points.items():
            c = colors[(numbering[qi] - 1) % len(colors)]
            ax.plot(x, y, "o", color=c, markersize=6, alpha=0.85)
            ax.annotate(str(numbering[qi]), (x, y), color=c, fontsize=9,
                        fontweight="bold", xytext=(0, 7),
                        textcoords="offset points")
        ax.axis("off")
    fig.suptitle(f"Общие ключевые точки (SIFT, эталон №{ref+1}, Ratio={ratio:.2f})",
                 fontsize=16)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "common_keypoints.png")
    fig.savefig(out, dpi=110)
    print(f"\nРисунок сохранён: {out}")


if __name__ == "__main__":
    main()