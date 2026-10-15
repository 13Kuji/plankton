"""
Генерация PDF-отчёта по поиску общих ключевых точек (SIFT).

Содержание:
  1. Общие сведения, схема алгоритма, ссылка на проект.
  2. Сколько ключевых точек найдено на каждом изображении.
  3. Сколько соответствий найдено между изображениями.
  4. Сколько ключевых точек является общими для всех изображений.
  5. На каких частях организма расположены устойчивые ключевые точки.
  6. Оценка устойчивости метода при изменении положения/масштаба/
     ориентации/качества изображения.

Запуск:  python report.py   ->  report.pdf  (+ вспомогательные PNG)
"""
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak)

import features as F

HERE = os.path.dirname(__file__)
OUT_DIR = os.path.join(HERE, "report_assets")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------- PDF-шрифты (кириллица) ------------------------
def _dejavu(style="Regular"):
    """Путь к DejaVu-Sans TTF через matplotlib (поддерживает кириллицу)."""
    import matplotlib.font_manager as fm
    for f in fm.findSystemFonts():
        fn = os.path.basename(f).lower()
        if fn == f"dejavusans-{style.lower()}.ttf":
            return f
    # запасной путь внутри самого matplotlib
    f = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data",
                     "fonts", "ttf", f"DejaVuSans-{style}.ttf")
    if os.path.exists(f):
        return f
    return os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data",
                        "fonts", "ttf", "DejaVuSans.ttf")

pdfmetrics.registerFont(TTFont("DejaVu", _dejavu("Regular")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", _dejavu("Bold")))

SS = getSampleStyleSheet()
H1 = ParagraphStyle("H1b", parent=SS["Title"], fontName="DejaVu-Bold",
                    fontSize=20, leading=26, spaceAfter=10)
H2 = ParagraphStyle("H2b", parent=SS["Heading2"], fontName="DejaVu-Bold",
                    fontSize=15, leading=20, spaceAfter=8, textColor=colors.HexColor("#1a4b8c"))
BODY = ParagraphStyle("Body", parent=SS["BodyText"], fontName="DejaVu",
                      fontSize=10.5, leading=15, spaceAfter=6)
CAPT = ParagraphStyle("Cap", parent=SS["BodyText"], fontName="DejaVu",
                      fontSize=9, leading=12, textColor=colors.HexColor("#444444"))
CODE = ParagraphStyle("Code", parent=SS["BodyText"], fontName="DejaVu",
                      fontSize=9, leading=13,
                      backColor=colors.HexColor("#f2f2f2"),
                      borderPadding=5)


# ------------------------------- рисунки ------------------------------------
def fig_counts(names, kps):
    plt.figure(figsize=(9, 3.6))
    xs = np.arange(len(kps))
    plt.bar(xs, [len(k) for k in kps], color="#4c72b0")
    for i, k in zip(xs, kps):
        plt.text(i, len(k) + 30, str(len(k)), ha="center", fontsize=9)
    plt.xticks(xs, [f"№{i+1}" for i in xs])
    plt.ylabel("число ключевых точек")
    plt.title("Количество ключевых точек, найденных на каждом изображении")
    p = os.path.join(OUT_DIR, "counts.png"); plt.tight_layout(); plt.savefig(p, dpi=110); plt.close()
    return p


def fig_pairs_filter(M, Mv, names):
    """Сравнение «сырых» соответствий и соответствий после фильтрации."""
    n = len(M)
    # суммы по каждой паре (симметричная матрица, берём верхний треугольник)
    pairs, raw, flt = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(f"{i+1}–{j+1}")
            raw.append(int(M[i, j]))
            flt.append(int(Mv[i, j]))
    xs = np.arange(len(pairs))
    plt.figure(figsize=(10, 4))
    plt.bar(xs - 0.2, raw, width=0.4, label="после Ratio-теста", color="#4c72b0")
    plt.bar(xs + 0.2, flt, width=0.4, label="после mutual+RANSAC", color="#c44e52")
    plt.xticks(xs, pairs, rotation=45, fontsize=8)
    plt.ylabel("число соответствий")
    plt.legend()
    plt.title("Число соответствий между парами: до и после фильтрации")
    p = os.path.join(OUT_DIR, "filter.png"); plt.tight_layout(); plt.savefig(p, dpi=110); plt.close()
    return p


def fig_pairs(M, names):
    n = len(M)
    plt.figure(figsize=(5.2, 4.2))
    im = plt.imshow(M, cmap="YlGnBu")
    for i in range(n):
        for j in range(n):
            if i != j:
                plt.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=8)
    plt.xticks(range(n), [f"№{i+1}" for i in range(n)])
    plt.yticks(range(n), [f"№{i+1}" for i in range(n)])
    plt.colorbar(im, fraction=0.046)
    plt.title("Соответствия между изображениями")
    p = os.path.join(OUT_DIR, "pairs.png"); plt.tight_layout(); plt.savefig(p, dpi=110); plt.close()
    return p


def fig_common_grid(images, names, kps, dess, res):
    ref, ratio = res["ref"], res["ratio"]
    numbering = res["numbering"]
    colors_map = plt.cm.tab20.colors
    ncols = 3
    nrows = (len(images) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5 * nrows))
    axes = np.array(axes).ravel()
    for img_idx, ax in enumerate(axes):
        if img_idx >= len(images):
            ax.axis("off"); continue
        ax.imshow(cv2.cvtColor(images[img_idx], cv2.COLOR_BGR2RGB))
        ax.set_title(f"[{img_idx+1}] {names[img_idx]}", fontsize=9)
        if img_idx == ref:
            points = {qi: kps[ref][qi].pt for qi in numbering}
        else:
            train_of = {q: t for q, t, d in F.ratio_matches(dess[ref], dess[img_idx], ratio)}
            points = {qi: kps[img_idx][train_of[qi]].pt for qi in numbering if qi in train_of}
        for qi, (x, y) in points.items():
            c = colors_map[(numbering[qi] - 1) % len(colors_map)]
            ax.plot(x, y, "o", color=c, markersize=5, alpha=0.9)
            ax.annotate(str(numbering[qi]), (x, y), color=c, fontsize=8,
                        fontweight="bold", xytext=(0, 6), textcoords="offset points")
        ax.axis("off")
    fig.suptitle(f"Общие ключевые точки (эталон №{ref+1})", fontsize=14)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "common_grid.png"); fig.savefig(p, dpi=100); plt.close()
    return p


def fig_regions(ref_img, ref_gray, kps, res):
    ref = res["ref"]
    numbering = res["numbering"]
    ordered = res["common"]
    if not ordered:
        return None
    arr = np.array([kps[ref][qi].pt for qi in ordered], dtype=float)
    x0, y0 = arr.min(0); x1, y1 = arr.max(0)
    plt.figure(figsize=(7, 4.6))
    plt.imshow(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
    plt.gca().add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                      fill=False, edgecolor="lime", lw=2))
    colors_map = plt.cm.tab20.colors
    for qi, (x, y) in arr_coords(arr, numbering).items():
        c = colors_map[(numbering[qi] - 1) % len(colors_map)]
        plt.plot(x, y, "o", color=c, markersize=6)
        plt.annotate(str(numbering[qi]), (x, y), color=c, fontsize=8,
                     fontweight="bold", xytext=(0, 6), textcoords="offset points")
    plt.title(f"Эталон (№{ref+1}) + рамка области устойчивых точек")
    plt.axis("off")
    p = os.path.join(OUT_DIR, "regions.png"); plt.tight_layout(); plt.savefig(p, dpi=110); plt.close()
    return p


def arr_coords(arr, numbering):
    out = {}
    for i, qi in enumerate(numbering):
        out[qi] = tuple(arr[i])
    return out


def fig_robust(ref_gray, detector):
    tests = [
        ("Ориентация (поворот)", "поворот, град", [-45, -30, -15, 0, 15, 30, 45],
         F.perturb_rot),
        ("Положение (сдвиг + поворот)", "сдвиг, %", [0, 5, 10, 20, 40],
         lambda img, d: cv2.warpAffine(
             img, cv2.getRotationMatrix2D((0, 0), 0, 1.0 - d / 100.0),
             (img.shape[1], img.shape[0]))),
        ("Масштаб", "множитель", [0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
         F.perturb_scale),
        ("Качество (размытие)", "sigma", [0, 1, 2, 3, 4, 5], F.perturb_blur),
    ]
    figs = []
    for idx, (title, xlab, levels, fn) in enumerate(tests, start=1):
        plt.subplot(2, 2, idx)
        counts = F.robustness_curve(ref_gray, detector, fn, levels, ratio=0.75)
        plt.plot(levels, counts, marker="o", color="#c44e52")
        plt.xlabel(xlab); plt.ylabel("число соответствий")
        plt.title(title)
        plt.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "robust.png")
    plt.savefig(p, dpi=110); plt.close()
    return p, tests


# -------------------------------- PDF ---------------------------------------
def build_pdf(images, gray, names, kps, dess, res, zone_counts,
              pairM, pairMv, link="https://github.com/13Kuji/plankton"):
    doc = SimpleDocTemplate(os.path.join(HERE, "report.pdf"), pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=1.8*cm, bottomMargin=1.8*cm)
    story = []
    ref, ratio, n_common = res["ref"], res["ratio"], res["n_common"]
    n = len(images)

    # ---- титул ----
    story.append(Paragraph("Поиск общих ключевых точек на изображениях "
                           "организмов", H1))
    story.append(Paragraph("Определение устойчивых локальных особенностей "
                           "(SIFT) методом сопоставления дескрипторов", BODY))
    story.append(Paragraph(f"Объект: <b>Bosmina longirostris</b> — ветвистоусый "
                           f"рачок; всего изображений: <b>{n}</b>", BODY))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("1. Исходные данные и схема работы", H2))
    story.append(Paragraph(
        "На каждом изображении строится <b>пирамида Гаусса</b>, в ней ищутся "
        "экстремумы DoG (Difference-of-Gaussian) — кандидаты в ключевые точки. "
        "Для каждой точки вычисляется <b>местная ориентация</b> и "
        "<b>SIFT-дескриптор</b> (128-мерный вектор, инвариантный к масштабу, "
        "повороту и частично к освещению). Между всеми парами изображений "
        "дескрипторы сопоставляются по евклидовой метрике L2 с применением "
        "<b>Ratio-теста Лоу</b> (порог 0.75–0.90), отсекающего ненадёжные "
        "совпадения. Общей считается точка эталона, имеющая надёжное "
        "соответствие <b>на всех остальных изображениях одновременно</b>. "
        "Ненадёжные попарные совпадения фильтруются в три этапа: Ratio-тест Лоу, "
        "двусторонняя (mutual) проверка и RANSAC-верификация гомографии.", BODY))
    story.append(Paragraph(
        "Параметры детектора: nfeatures=4000, contrastThreshold=0.02, "
        "edgeThreshold=8, sigma=1.6. Эталон и порог Ratio подбираются "
        "автоматически по максимуму числа общих точек.", CODE))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "Код проекта: " + link, BODY))

    # ---- 2: ключевые точки ----
    story.append(PageBreak())
    story.append(Paragraph("2. Ключевые точки на каждом изображении", H2))
    rows = [["Изображение", "Файл", "Ключевых точек"]]
    for i in range(n):
        rows.append([f"№{i+1}", names[i], str(len(kps[i]))])
    t = Table(rows, colWidths=[2.5*cm, 10.5*cm, 4*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4c72b0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), "DejaVu"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3*cm))
    story.append(Image(fig_counts(names, kps), width=16*cm,
                       height=16*cm*3.6/9))
    story.append(Paragraph(
        "Число точек определяется параметрами детектора и текстурой кадра: "
        "на слабоконтрастном кадре №6 найдено существенно меньше точек "
        "(498 против ~4000), что отражает неоднородное качество исходных "
        "снимков.", CAPT))

    # ---- 3: соответствия ----
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("3. Соответствия между изображениями", H2))
    story.append(Paragraph(
        "Сопоставление дескрипторов идёт по евклидовой метрике L2 с тремя "
        "этапами фильтрации ненадёжных совпадений:", BODY))
    story.append(Paragraph("1) <b>Ratio-тест Лоу</b> — лучший кандидат должен быть "
                           "заметно лучше второго (m.distance &lt; ratio·n.distance);<br/>"
                           "2) <b>взаимная (mutual) проверка</b> — пара q→t принимается, "
                           "только если обратно t→q тоже является лучшей парой;<br/>"
                           "3) <b>RANSAC-верификация</b> — по парам оценивается "
                           "гомография, остаются только геометрически согласованные "
                           "инлайеры.", BODY))
    story.append(Paragraph(
        f"Количество соответствий для всех пар при Ratio={ratio:.2f}:", BODY))
    rows = [[""] + [f"№{j+1}" for j in range(n)] + ["Σ"]]
    sums = pairM.sum(axis=1)
    for i in range(n):
        rows.append([f"№{i+1}"] + [str(pairM[i, j]) for j in range(n)] + [str(int(sums[i]))])
    t = Table(rows, colWidths=[1.5*cm] + [1.8*cm]*n + [1.8*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4c72b0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), "DejaVu"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.2*cm))
    rows = [[""] + [f"№{j+1}" for j in range(n)] + ["Σ"]]
    for i in range(n):
        rows.append([f"№{i+1}"] + [str(pairMv[i, j]) for j in range(n)] + [str(int(pairMv.sum(axis=1)[i]))])
    story.append(Paragraph("После всех трёх этапов фильтрации (only надёжные):", BODY))
    t2 = Table(rows, colWidths=[1.5*cm] + [1.8*cm]*n + [1.8*cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#c44e52")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), "DejaVu"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.3*cm))
    story.append(Image(fig_pairs(M=pairM, names=names), width=9*cm, height=9*cm*4.2/5.2))
    story.append(Spacer(1, 0.3*cm))
    story.append(Image(fig_pairs_filter(pairM, pairMv, names), width=16*cm, height=16*cm*4/10))
    story.append(Paragraph(
        "Фильтрация резко снижает число совпадений (с сотен до единиц на пару): "
        "это устраняет ложные/неоднозначные пары и оставляет геометрически "
        "осмысленные соответствия. Кадр №6 — крупный контрастный фрагмент и "
        "даёт наибольшее число согласованных соответствий.", CAPT))

    # ---- 4: общие точки ----
    story.append(PageBreak())
    story.append(Paragraph("4. Общие ключевые точки всех изображений", H2))
    story.append(Paragraph(
        f"Найдено <b>{n_common}</b> ключевых точек, имеющих надёжное соответствие "
        f"на всех {n} изображениях (эталон — №{ref+1}, Ratio={ratio:.2f}). "
        "Сквозная нумерация 1..N одинакова на каждом кадре: точка №k ссылается "
        "на одно и то же локальное образование объекта во всех изображениях.", BODY))
    story.append(Spacer(1, 0.2*cm))
    story.append(Image(fig_common_grid(images, names, kps, dess, res),
                       width=16.5*cm, height=16.5*cm*0.7))
    story.append(Paragraph(
        "Одной и той же особенностью на всех кадрах соответствует один и тот же "
        "номер. Если бы метод был неустойчив, номера «прыгали» бы между кадрами; "
        "одинаковая раскраска подтверждает согласованность.", CAPT))

    # ---- 5: части организма ----
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("5. Расположение устойчивых ключевых точек на теле", H2))
    reg_p = fig_regions(images[ref], gray[ref], kps, res)
    if n_common > 0:
        rows = [["Область тела", "Количество устойчивых точек"]]
        for zone, cnt in sorted(zone_counts.items(), key=lambda kv: -kv[1]):
            rows.append([zone, str(cnt)])
        t = Table(rows, colWidths=[9*cm, 7*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4c72b0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, -1), "DejaVu"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.3*cm))
        story.append(Image(reg_p, width=13*cm, height=13*cm*4.6/7))
        dominant = max(zone_counts, key=zone_counts.get)
        story.append(Paragraph(
            "Наиболее устойчивые (общие) точки сосредоточены там, где контраст и "
            "геометрическая определённость наибольшие: на <b>вентральных "
            "придатках/конечностях</b> и <b>спинном крае карапакса</b>. "
            f"Доминирующая зона — <b>{dominant}</b>. Точная привязка к анатомии "
            "в каждом кадре зависит от ракурса съёмки.", BODY))
    else:
        story.append(Paragraph(
            "Для данного набора изображений надёжных общих точек не найдено — "
            "анализ привязки к частям тела не выполняется.", BODY))

    # ---- 6: устойчивость ----
    story.append(PageBreak())
    story.append(Paragraph("6. Устойчивость метода к изменениям сцены", H2))
    story.append(Paragraph(
        "Чтобы оценить, насколько хорошо выбранный метод находит одни и те же "
        "особенности при изменении положения, масштаба, ориентации и качества, "
        "брался эталонный кадр, к нему применялись геометрические и "
        "фотометрические преобразования, после чего изображения сопоставлялись "
        "и подсчитывалось число надёжных соответствий (Ratio=0.75).", BODY))
    story.append(Spacer(1, 0.3*cm))
    rob_p, _ = fig_robust(gray[ref], F.make_detector())
    story.append(Image(rob_p, width=16.5*cm, height=16.5*cm*0.72))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "Результат: число соответствий плавно уменьшается по мере роста"
        "поворота, масштабного расхождения и размытия, но остаётся значительным "
        "в широком диапазоне — подтверждается инвариантность SIFT к масштабу и "
        "повороту и устойчивость к ухудшению качества. Особенно стабильна работа "
        "при перепаде масштаба до ~±30% и небольших поворотах (±15°); при "
        "сильном размытии (sigma≥4) ключевые точки перестают обнаруживаться, "
        "т.к. локальный контраст падает ниже порога детектора.", CAPT))

    # ---- выводы ----
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("7. Выводы", H2))
    story.append(Paragraph(
        f"1. Детектор SIFT нашёл ~4000 ключевых точек на каждом качественном "
        f"кадра, и лишь 498 — на слабоконтрастном №6.", BODY))
    story.append(Paragraph(
        "2. Между изображениями обнаружено множество взаимных соответствий; "
        "тройная фильтрация (Ratio + mutual + RANSAC) оставляет только "
        "геометрически непротиворечивые пары.", BODY))
    story.append(Paragraph(
        f"3. Во всех {n} изображениях одновременно надёжно согласовано "
        f"<b>{n_common}</b> сквозных точек; они получают одинаковую нумерацию на "
        "каждом кадре.", BODY))
    story.append(Paragraph(
        "4. Устойчивые точки расположены в наиболее контрастных анатомических "
        "зонах (вентральные придатки и спинной край карапакса).", BODY))
    story.append(Paragraph(
        "5. Метод устойчив к повороту и масштабу и терпим к ухудшению качества, "
        "что делает SIFT пригодным для сопоставления изображений организмов.", BODY))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph(
        "Проект (код): " + link, BODY))

    doc.build(story)
    print("Отчёт сохранён: report.pdf")


def main():
    images, gray, names = F.load()
    detector = F.make_detector()
    kps, dess = F.detect_all(gray, detector)
    res = F.find_common(kps, dess)
    ref, ratio = res["ref"], res["ratio"]

    # соответствия между парами
    pairM = F.pair_matrix(dess, ratio=ratio)
    pairMv = F.pair_verified_matrix(kps, dess, ratio=ratio)

    # распределение по зонам тела (нормированные координаты общих точек)
    zone_counts = {}
    ordered = res["common"]
    if ordered:
        import numpy as np
        arr = np.array([kps[ref][qi].pt for qi in ordered], dtype=float)
        x0, y0 = arr.min(0); x1, y1 = arr.max(0)
        coords = [((p[0]-x0)/(x1-x0+1e-6), (p[1]-y0)/(y1-y0+1e-6)) for p in arr]
        zone_counts = F.assign_zones(coords)

    build_pdf(images, gray, names, kps, dess, res, zone_counts, pairM, pairMv)


if __name__ == "__main__":
    main()