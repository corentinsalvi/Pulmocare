"""
dicom_seg_viewer.py v3 — CT + Segmentation tumeur (overlay + 3D)
Corrections :
  - Bug matplotlib 3.8+ : QuadContourSet.collections supprimé → remplacé par imshow RGBA
  - Alignement SEG par position Z (fix QIN-LungCT-Seg qui n'utilise pas les SOPInstanceUIDs)

Usage : python dicom_seg_viewer.py "chemin/vers/dossier_patient/"
Require : pip install pydicom numpy scipy scikit-image matplotlib
"""

import os, sys, warnings
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider, Button
from scipy import ndimage
from skimage import measure

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════
#  1. DÉTECTION CT / SEG
# ══════════════════════════════════════════════════════════════

def find_ct_and_seg(root_folder):
    ct_files, seg_files = [], []
    for root, _, files in os.walk(root_folder):
        for f in files:
            if not f.lower().endswith(".dcm"):
                continue
            path = os.path.join(root, f)
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True)
                sop = str(getattr(ds, "SOPClassUID", ""))
                mod = getattr(ds, "Modality", "")
                if "66.4" in sop or mod == "SEG":
                    seg_files.append(path)
                elif mod == "CT":
                    ct_files.append(path)
            except Exception:
                pass
    if not ct_files:
        raise FileNotFoundError("Aucune série CT trouvée.")
    if not seg_files:
        raise FileNotFoundError("Aucun SEG trouvé.\n→ Télécharge 'CT Images & Segmentations Combined' sur TCIA.")
    print(f"[OK] CT  : {len(ct_files)} fichiers")
    print(f"[OK] SEG : {len(seg_files)} fichier(s)")
    return ct_files, seg_files


# ══════════════════════════════════════════════════════════════
#  2. CHARGEMENT CT
# ══════════════════════════════════════════════════════════════

def load_ct(ct_files):
    slices = []
    for path in ct_files:
        try:
            ds = pydicom.dcmread(path)
            if hasattr(ds, "pixel_array"):
                slices.append(ds)
        except Exception:
            pass
    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except AttributeError:
        try:
            slices.sort(key=lambda s: int(s.InstanceNumber))
        except AttributeError:
            pass

    volume = np.stack([
        s.pixel_array.astype(np.float32)
        * float(getattr(s, "RescaleSlope", 1))
        + float(getattr(s, "RescaleIntercept", -1024))
        for s in slices
    ])
    try:
        ps = slices[0].PixelSpacing
        sy, sx = float(ps[0]), float(ps[1])
    except AttributeError:
        sy, sx = 1.0, 1.0
    try:
        sz = abs(float(slices[1].ImagePositionPatient[2]) - float(slices[0].ImagePositionPatient[2]))
    except (AttributeError, IndexError):
        sz = float(getattr(slices[0], "SliceThickness", 1.0))

    sop_uids = [str(s.SOPInstanceUID) for s in slices]
    print(f"[OK] Volume CT : {volume.shape}  |  spacing {sz:.2f}×{sy:.2f}×{sx:.2f} mm")
    return volume, (sz, sy, sx), sop_uids, slices


# ══════════════════════════════════════════════════════════════
#  3. CHARGEMENT SEG — alignement par Z (fix QIN-LungCT-Seg)
# ══════════════════════════════════════════════════════════════

def load_seg_mask(seg_files, sop_uids, volume_shape, ct_slices):
    n_z, n_y, n_x = volume_shape
    mask = np.zeros(volume_shape, dtype=np.uint8)
    uid_to_idx = {uid: i for i, uid in enumerate(sop_uids)}

    # Index Z → coupe
    z_to_idx = {}
    for i, s in enumerate(ct_slices):
        try:
            z_to_idx[round(float(s.ImagePositionPatient[2]), 2)] = i
        except AttributeError:
            pass
    z_sorted = sorted(z_to_idx.keys())

    def nearest_z(z_val, tol=2.0):
        if not z_sorted:
            return None
        closest = min(z_sorted, key=lambda k: abs(k - z_val))
        return z_to_idx[closest] if abs(closest - z_val) <= tol else None

    for seg_path in seg_files:
        try:
            ds = pydicom.dcmread(seg_path)
        except Exception as e:
            print(f"[SKIP] {e}")
            continue

        pixel_data = ds.pixel_array
        if pixel_data.ndim == 2:
            pixel_data = pixel_data[np.newaxis]
        n_frames = pixel_data.shape[0]
        aligned = 0

        # Stratégie 1 : PerFrameFunctionalGroupsSequence
        if hasattr(ds, "PerFrameFunctionalGroupsSequence"):
            for fi, frame in enumerate(ds.PerFrameFunctionalGroupsSequence):
                if fi >= n_frames:
                    break
                ct_idx = None
                # 1a. SOPInstanceUID
                try:
                    ref_uid = str(frame.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID)
                    ct_idx = uid_to_idx.get(ref_uid)
                except (AttributeError, IndexError):
                    pass
                # 1b. Position Z
                if ct_idx is None:
                    try:
                        z = round(float(frame.PlanePositionSequence[0].ImagePositionPatient[2]), 2)
                        ct_idx = z_to_idx.get(z) if z in z_to_idx else nearest_z(z)
                    except (AttributeError, IndexError):
                        pass
                if ct_idx is not None:
                    fd = pixel_data[fi]
                    if fd.shape != (n_y, n_x):
                        from scipy.ndimage import zoom as ndzoom
                        fd = ndzoom(fd, (n_y / fd.shape[0], n_x / fd.shape[1]), order=0)
                    mask[ct_idx] = np.maximum(mask[ct_idx], fd.astype(np.uint8))
                    aligned += 1

        # Stratégie 2 : ReferencedSeriesSequence
        elif hasattr(ds, "ReferencedSeriesSequence"):
            try:
                refs = ds.ReferencedSeriesSequence[0].ReferencedInstanceSequence
                for i, ref in enumerate(refs):
                    if i >= n_frames:
                        break
                    ct_idx = uid_to_idx.get(str(ref.ReferencedSOPInstanceUID))
                    if ct_idx is not None:
                        mask[ct_idx] = np.maximum(mask[ct_idx], pixel_data[i].astype(np.uint8))
                        aligned += 1
            except (AttributeError, IndexError):
                pass

        # Stratégie 3 : empilement direct
        if aligned == 0:
            print(f"[WARN] Empilement direct ({n_frames} frames)")
            for i in range(min(n_frames, n_z)):
                mask[i] = np.maximum(mask[i], pixel_data[i].astype(np.uint8))

    n_vox = mask.sum()
    if n_vox == 0:
        print("[ATTENTION] Masque vide — SEG non aligné sur le CT.")
    else:
        print(f"[OK] Masque : {n_vox:,} voxels | {mask.any(axis=(1,2)).sum()} coupe(s) avec tumeur")
    return mask


# ══════════════════════════════════════════════════════════════
#  4. STATS TUMEUR
# ══════════════════════════════════════════════════════════════

def tumor_stats(volume, mask, spacing):
    sz, sy, sx = spacing
    n_vox = int(mask.sum())
    if n_vox == 0:
        return {}
    vv = sz * sy * sx
    hu = volume[mask.astype(bool)]
    z_idx = np.where(mask.any(axis=(1, 2)))[0]
    return {
        "volume_cm3":  round(n_vox * vv / 1000, 2),
        "diameter_mm": round(2 * ((3 * n_vox * vv) / (4 * np.pi)) ** (1/3), 1),
        "hu_mean":     round(float(hu.mean()), 1),
        "hu_min":      round(float(hu.min()), 1),
        "hu_max":      round(float(hu.max()), 1),
        "z_first":     int(z_idx[0]),
        "z_last":      int(z_idx[-1]),
        "z_center":    int(z_idx[len(z_idx) // 2]),
        "n_slices":    len(z_idx),
    }


# ══════════════════════════════════════════════════════════════
#  5. VUE 3D
# ══════════════════════════════════════════════════════════════

def show_3d(mask, spacing, stats):
    if not mask.any():
        print("[INFO] Masque vide.")
        return
    from scipy.ndimage import gaussian_filter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    smooth = gaussian_filter(mask.astype(np.float32), sigma=1.0)
    try:
        verts, faces, _, _ = measure.marching_cubes(smooth, level=0.5, spacing=spacing)
    except Exception as e:
        print(f"[ERREUR] {e}")
        return
    fig = plt.figure(figsize=(9, 8))
    fig.patch.set_facecolor("#0d0d0d")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0d0d0d")
    mesh = Poly3DCollection(verts[faces], alpha=0.85, linewidths=0)
    mesh.set_facecolor("#e53935")
    mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)
    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())
    for p in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        p.fill = False
        p.set_edgecolor("#333")
    ax.tick_params(colors="gray", labelsize=7)
    ax.set_title(
        f"Vue 3D  |  ⌀ {stats.get('diameter_mm','?')} mm  |  Vol. {stats.get('volume_cm3','?')} cm³",
        color="white", fontsize=11, pad=12)
    fig.text(0.5, 0.02, "Cliquer-glisser pour faire tourner", ha="center", color="#555", fontsize=8)
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════
#  6. VIEWER 2D  — fix matplotlib 3.8+ : imshow RGBA au lieu de contour
# ══════════════════════════════════════════════════════════════

def launch_viewer(volume, mask, spacing, stats):
    n = volume.shape[0]
    current_idx = [stats.get("z_center", n // 2)]
    windows = {"Poumon": (-600, 1500), "Médiastin": (40, 400), "Os": (400, 1800)}
    win_names = list(windows.keys())
    current_win = [0]
    show_overlay = [True]
    has_mask = bool(mask.any())

    def get_display(idx):
        wc, ww = windows[win_names[current_win[0]]]
        lo, hi = wc - ww / 2, wc + ww / 2
        return (np.clip(volume[idx], lo, hi) - lo) / (hi - lo)

    def make_rgba(mask_slice):
        """Masque rouge semi-transparent — compatible toutes versions matplotlib."""
        rgba = np.zeros((*mask_slice.shape, 4), dtype=np.float32)
        rgba[..., 0] = 1.0   # R
        rgba[..., 1] = 0.09  # G
        rgba[..., 2] = 0.09  # B
        rgba[..., 3] = np.where(mask_slice > 0, 0.50, 0.0)  # alpha
        return rgba

    # Layout
    fig = plt.figure(figsize=(13, 9))
    fig.patch.set_facecolor("#0d0d0d")
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    gs.update(left=0.03, right=0.97, bottom=0.15, top=0.93, wspace=0.06)
    ax      = fig.add_subplot(gs[0])
    ax_info = fig.add_subplot(gs[1])
    ax.set_facecolor("#0d0d0d")
    ax_info.set_facecolor("#111")
    ax.axis("off")
    ax_info.axis("off")

    im_ct   = ax.imshow(get_display(current_idx[0]), cmap="gray",
                        interpolation="bilinear", zorder=0)
    im_mask = ax.imshow(make_rgba(mask[current_idx[0]]),
                        interpolation="nearest", zorder=1)

    title = ax.set_title("", color="white", fontsize=11, pad=8)

    ax_info.set_title("Tumeur", color="#aaa", fontsize=10, pad=6)
    info_str = (
        f"Diamètre\n  {stats.get('diameter_mm','—')} mm\n\n"
        f"Volume\n  {stats.get('volume_cm3','—')} cm³\n\n"
        f"HU moyen\n  {stats.get('hu_mean','—')}\n\n"
        f"HU min/max\n  {stats.get('hu_min','—')} / {stats.get('hu_max','—')}\n\n"
        f"Coupes\n  {stats.get('z_first','—')} → {stats.get('z_last','—')}\n"
        f"  ({stats.get('n_slices','—')} coupes)\n\n"
        f"Centre\n  coupe {stats.get('z_center','—')}"
    ) if stats else "Aucune segmentation\ndisponible"
    ax_info.text(0.08, 0.95, info_str, transform=ax_info.transAxes,
                 color="white", fontsize=9.5, va="top", family="monospace", linespacing=1.6)

    def draw_frame(idx):
        im_ct.set_data(get_display(idx))
        if has_mask and show_overlay[0]:
            im_mask.set_data(make_rgba(mask[idx]))
            im_mask.set_visible(True)
        else:
            im_mask.set_visible(False)
        on_tumor = has_mask and bool(mask[idx].any())
        title.set_text(
            f"Coupe {idx+1}/{n}{'  🔴 TUMEUR' if on_tumor else ''}  |  "
            f"Fenêtre : {win_names[current_win[0]]}"
        )
        fig.canvas.draw_idle()

    draw_frame(current_idx[0])

    ax.legend(handles=[mpatches.Patch(color="#ff1744", alpha=0.6, label="Segmentation tumeur")],
              loc="upper left", facecolor="#1a1a1a", edgecolor="#444",
              labelcolor="white", fontsize=8, framealpha=0.85)

    # Slider
    ax_sl = plt.axes([0.04, 0.07, 0.62, 0.028], facecolor="#1e1e1e")
    slider = Slider(ax_sl, "Coupe", 0, n - 1, valinit=current_idx[0], valstep=1, color="#e53935")
    slider.label.set_color("white")
    slider.valtext.set_color("white")

    def on_slider(val):
        idx = int(slider.val)
        current_idx[0] = idx
        draw_frame(idx)
    slider.on_changed(on_slider)

    # Bouton 3D
    ax_b3d = plt.axes([0.70, 0.06, 0.12, 0.045])
    btn_3d = Button(ax_b3d, "Vue 3D 🫁", color="#1a1a1a", hovercolor="#333")
    btn_3d.label.set_color("white")
    btn_3d.on_clicked(lambda e: (plt.close("all"), show_3d(mask, spacing, stats)))

    # Bouton overlay
    ax_bov = plt.axes([0.84, 0.06, 0.12, 0.045])
    btn_ov = Button(ax_bov, "Overlay ON", color="#1a1a1a", hovercolor="#333")
    btn_ov.label.set_color("#ff1744")

    def on_overlay(e):
        show_overlay[0] = not show_overlay[0]
        btn_ov.label.set_text("Overlay ON" if show_overlay[0] else "Overlay OFF")
        btn_ov.label.set_color("#ff1744" if show_overlay[0] else "#555")
        draw_frame(current_idx[0])
    btn_ov.on_clicked(on_overlay)

    def on_scroll(event):
        idx = current_idx[0]
        slider.set_val(min(idx + 1, n - 1) if event.button == "up" else max(idx - 1, 0))

    def on_key(event):
        idx = current_idx[0]
        if event.key in ("right", "up"):      idx = min(idx + 1, n - 1)
        elif event.key in ("left", "down"):   idx = max(idx - 1, 0)
        elif event.key == "n":
            f = [z for z in range(idx + 1, n) if mask[z].any()]
            if f: idx = f[0]
        elif event.key == "p":
            p = [z for z in range(idx - 1, -1, -1) if mask[z].any()]
            if p: idx = p[0]
        elif event.key == "w":
            current_win[0] = (current_win[0] + 1) % len(win_names)
        slider.set_val(idx)

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.suptitle("🫁  CT + Segmentation Tumeur", color="white", fontsize=13, fontweight="bold", y=0.98)
    fig.text(0.5, 0.01, "Molette / ← →  |  N/P = coupe tumeur suiv./préc.  |  W = fenêtrage",
             ha="center", color="#444", fontsize=8.5)
    plt.show()


# ══════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        folder = input("Chemin vers le dossier patient : ").strip()
    else:
        folder = sys.argv[1]

    if not os.path.isdir(folder):
        print(f"[ERREUR] Dossier introuvable : {folder}")
        sys.exit(1)

    ct_files, seg_files = find_ct_and_seg(folder)
    volume, spacing, sop_uids, slices = load_ct(ct_files)
    mask  = load_seg_mask(seg_files, sop_uids, volume.shape, slices)
    stats = tumor_stats(volume, mask, spacing)

    if stats:
        print(f"\n  ⌀ {stats['diameter_mm']} mm  |  Vol. {stats['volume_cm3']} cm³  |  "
              f"HU moy. {stats['hu_mean']}  |  Coupes {stats['z_first']}→{stats['z_last']}\n")

    launch_viewer(volume, mask, spacing, stats)
