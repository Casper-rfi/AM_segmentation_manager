# Reference: real dataset layouts

Ground-truth path examples collected from real data. This is a **reference**,
not a plan: it is what future layout work gets checked against, and it is the
evidence behind the conventions implemented in
[`layout.py`](../src/am_segmentation_manager/layout.py).

> **These are representative examples, not a definitive taxonomy.**
>
> The sections below are datasets that happen to have been looked at. They are
> not an exhaustive classification of what Adaptive Milling can produce, and the
> numbering is for reference within this document — there is no guarantee that a
> new dataset falls into one of these groups, or into exactly one.
>
> What actually varies is a set of **independent axes**:
>
> | Axis | Values seen |
> |---|---|
> | Lamella level | present, or absent with the project standing in |
> | Stage level | present, or absent with the lamella standing in |
> | Stage nesting | direct child, or under `Milling/Polishing/` |
> | Stage naming | `adaptive_polish`, or with a timestamp appended |
> | Image directory naming | `sem`/`fib`, `SEM`/`FIB`, `SEM_8` |
> | Label directory naming | `labels`/`labels_fib`, `labels_FIB`, `<image-dir>_labels` |
> | Imaging-condition variants | absent, or several sharing one label |
> | Modalities present | both, or SEM only |
>
> **These combine freely.** Modality presence in particular is a property of a
> *dataset*, not of its layout: §4 and §5 below have no FIB, but §6 — an older
> layout than either — has FIB throughout, and every lamella of the oldest
> dataset in the current working collection has FIB images.
>
> Tests should therefore vary the axis a behaviour depends on, rather than
> reaching for the example that happens to exhibit it. `tests/conftest.py`
> builds datasets that way, with these examples as named starting points.

All paths are relative to the Autolamella project, and show an example SEM and
FIB image with its corresponding label. How each of these layouts is resolved is
in [`decisions.md`](decisions.md) § Filesystem conventions.

Note that §4 and §5 are **not** in the current working collection at
`20260807_lamella_segmentations_new`; they are recorded here from elsewhere.

---

## 1. Current Autolamella projects (2026)

Lamella → `Milling/Polishing/` → timestamped stage → modality directory.

```
SEM   AutoLamella-2026-07-28-13-57/01-alive-muskox/Milling/Polishing/adaptive_polish_2026-07-28-02-43-32PM/sem/01-alive-muskox_AP_img_000_SEM.tif
label AutoLamella-2026-07-28-13-57/01-alive-muskox/Milling/Polishing/adaptive_polish_2026-07-28-02-43-32PM/labels/01-alive-muskox_AP_img_000_SEM.tif

FIB   AutoLamella-2026-07-28-13-57/01-alive-muskox/Milling/Polishing/adaptive_polish_2026-07-28-02-43-32PM/fib/01-alive-muskox_AP_img_000_FIB.tif
label AutoLamella-2026-07-28-13-57/01-alive-muskox/Milling/Polishing/adaptive_polish_2026-07-28-02-43-32PM/labels_fib/01-alive-muskox_AP_img_000_FIB.tif
```

## 2. Older Autolamella projects — stage without timestamp (2024)

Lamella → `adaptive_polish/` directly, no `Milling/Polishing/` level.

```
SEM   grid12-AutoLamella-2024-08-07-18-30/07-known-mutt/adaptive_polish/sem/adapt_mill_img_000_electron.tif
label grid12-AutoLamella-2024-08-07-18-30/07-known-mutt/adaptive_polish/labels/adapt_mill_img_000_electron.tif

FIB   grid12-AutoLamella-2024-08-07-18-30/07-known-mutt/adaptive_polish/fib/adapt_mill_img_000_electron.tif
label grid12-AutoLamella-2024-08-07-18-30/07-known-mutt/adaptive_polish/labels_fib/adapt_mill_img_000_electron.tif
```

```
SEM   AutoLamella-2024-10-07-10-50/09-modern-dove/adaptive_polish/sem/09-modern-dove_AP_img_000_SEM.tif
label AutoLamella-2024-10-07-10-50/09-modern-dove/adaptive_polish/labels/09-modern-dove_AP_img_000_SEM.tif

FIB   AutoLamella-2024-10-07-10-50/09-modern-dove/adaptive_polish/fib/09-modern-dove_AP_img_000_SEM.tif
label AutoLamella-2024-10-07-10-50/09-modern-dove/adaptive_polish/labels_fib/09-modern-dove_AP_img_000_SEM.tif
```

**Note the filenames.** The FIB image is named `..._SEM.tif` in this generation.
Filename suffix is therefore *not* a reliable modality marker; the directory is.

## 3. Older Autolamella projects — timestamped stage (2025)

```
SEM   AutoLamella-2025-04-14-20-54/06-kind-moray/adaptive_polish_2025-04-15-04-24-02AM/sem/06-kind-moray_AP_img_000_SEM.tif
label AutoLamella-2025-04-14-20-54/06-kind-moray/adaptive_polish_2025-04-15-04-24-02AM/labels/06-kind-moray_AP_img_000_SEM.tif

FIB   AutoLamella-2025-04-14-20-54/06-kind-moray/adaptive_polish_2025-04-15-04-24-02AM/fib/06-kind-moray_AP_img_000_SEM.tif
label AutoLamella-2025-04-14-20-54/06-kind-moray/adaptive_polish_2025-04-15-04-24-02AM/labels_fib/06-kind-moray_AP_img_000_SEM.tif
```

```
SEM   AutoLamella-2025-07-21-15-21_AM01/08-happy-shrimp/adaptive_polish_2025-07-22-12-54-52AM/sem/08-happy-shrimp_AP_img_000_SEM.tif
label AutoLamella-2025-07-21-15-21_AM01/08-happy-shrimp/adaptive_polish_2025-07-22-12-54-52AM/labels/08-happy-shrimp_AP_img_000_SEM.tif

FIB   AutoLamella-2025-07-21-15-21_AM01/08-happy-shrimp/adaptive_polish_2025-07-22-12-54-52AM/fib/08-happy-shrimp_AP_img_000_SEM.tif
label AutoLamella-2025-07-21-15-21_AM01/08-happy-shrimp/adaptive_polish_2025-07-22-12-54-52AM/labels_fib/08-happy-shrimp_AP_img_000_SEM.tif
```

## 4. Pre-Autolamella data — no lamella or stage directory

These have neither `experiment.yaml` nor `protocol.yaml`. This particular
dataset has no FIB images — a property of *it*, not of the collapsed layout;
compare §6, which is older still and has FIB throughout.

```
SEM   20231102/SEM/SEM_1.tif
label 20231102/labels/SEM_1.tif
```

## 5. Pre-Autolamella data — lamella directory, no stage directory

No FIB images in this dataset. See the note in §4.

```
SEM   20240223_adaptive_milling_ai/lamella_01/SEM/lamella01_09.tif
label 20240223_adaptive_milling_ai/lamella_01/labels/lamella01_09.tif
```

## 6. Pre-Autolamella data — multiple imaging conditions sharing one label

One acquisition captured under five SEM conditions, all resolving to the same
label file. This is the case that drives the "label is the unit of workflow
state" rule: moving that label without moving all five images would silently
change their state.

Note there is no unsuffixed `SEM/` directory here — the conditions are
`SEM_8`, `SEM_8_fi_1` and so on — so a filter for "the base directory" selects
none of them.

```
SEM (condition 1)  20240228_adaptive_milling_ai/lam_010/SEM_8/img_0004.tif
SEM (condition 2)  20240228_adaptive_milling_ai/lam_010/SEM_8_fi_1/img_0004.tif
SEM (condition 3)  20240228_adaptive_milling_ai/lam_010/SEM_8_fi_4/img_0004.tif
SEM (condition 4)  20240228_adaptive_milling_ai/lam_010/SEM_8_fi_8/img_0004.tif
SEM (condition 5)  20240228_adaptive_milling_ai/lam_010/SEM_8_fi_16/img_0004.tif
shared label       20240228_adaptive_milling_ai/lam_010/labels/img_0004.tif

FIB                20240228_adaptive_milling_ai/lam_010/FIB/img_0004.tif
label              20240228_adaptive_milling_ai/lam_010/labels_FIB/img_0004.tif
```
