"""
app.py — PulmoCare Flask + PyVista
Pipeline identique à pulmocare_pyvista.py,
maillages servis en JSON pour Three.js dans index.html

Usage : python app.py  →  http://localhost:5000
Install : pip install flask pyvista pydicom numpy scipy scikit-image
"""

import os, warnings, threading
import numpy as np
import pydicom
import pyvista as pv
from flask import Flask, render_template, request, jsonify
from scipy.ndimage import (binary_fill_holes, binary_closing,
                            binary_erosion, gaussian_filter, label)
from skimage import measure
from collections import defaultdict

warnings.filterwarnings("ignore")

app       = Flask(__name__)
_cache    = {}
_progress = {"pct": 0, "msg": "En attente…", "done": False, "error": None}


# ══════════════════════════════════════════════════════════════
#  DICOM
# ══════════════════════════════════════════════════════════════

def find_ct_and_seg(folder):
    ct, seg = [], []
    for root, _, files in os.walk(folder):
        for f in files:
            if not f.lower().endswith(".dcm"): continue
            path = os.path.join(root, f)
            try:
                ds  = pydicom.dcmread(path, stop_before_pixels=True)
                sop = str(getattr(ds, "SOPClassUID", ""))
                mod = getattr(ds, "Modality", "")
                if "66.4" in sop or mod == "SEG": seg.append(path)
                elif mod == "CT":                 ct.append(path)
            except: pass
    return ct, seg


def load_ct(ct_files):
    slices = []
    for p in ct_files:
        try:
            ds = pydicom.dcmread(p)
            if hasattr(ds, "pixel_array"): slices.append(ds)
        except: pass
    try:    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except: slices.sort(key=lambda s: int(getattr(s, "InstanceNumber", 0)))
    volume = np.stack([
        s.pixel_array.astype(np.float32) * float(getattr(s, "RescaleSlope", 1))
        + float(getattr(s, "RescaleIntercept", -1024))
        for s in slices])
    try:    ps = slices[0].PixelSpacing; sy, sx = float(ps[0]), float(ps[1])
    except: sy, sx = 1.0, 1.0
    try:    sz = abs(float(slices[1].ImagePositionPatient[2]) - float(slices[0].ImagePositionPatient[2]))
    except: sz = float(getattr(slices[0], "SliceThickness", 1.0))
    sop_uids = [str(s.SOPInstanceUID) for s in slices]
    print(f"[CT] {volume.shape} | {sz:.2f}x{sy:.2f}x{sx:.2f} mm")
    return volume, (sz, sy, sx), sop_uids, slices


def load_seg_mask(seg_files, sop_uids, shape, ct_slices):
    nz, ny, nx = shape
    uid_to_idx = {u: i for i, u in enumerate(sop_uids)}
    z_to_idx   = {}
    for i, s in enumerate(ct_slices):
        try: z_to_idx[round(float(s.ImagePositionPatient[2]), 2)] = i
        except: pass
    z_sorted = sorted(z_to_idx.keys())

    def nz_fn(z, tol=2.0):
        if not z_sorted: return None
        c = min(z_sorted, key=lambda k: abs(k - z))
        return z_to_idx[c] if abs(c - z) <= tol else None

    def align_one_seg(ds):
        px = ds.pixel_array
        if px.ndim == 2: px = px[np.newaxis]
        nf = px.shape[0]
        seg_masks = {}
        if hasattr(ds, "PerFrameFunctionalGroupsSequence"):
            for fi, frame in enumerate(ds.PerFrameFunctionalGroupsSequence):
                if fi >= nf: break
                seg_num = 1
                try: seg_num = int(frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
                except: pass
                if seg_num not in seg_masks:
                    seg_masks[seg_num] = np.zeros(shape, dtype=np.uint8)
                ci = None
                try:
                    uid = str(frame.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID)
                    ci  = uid_to_idx.get(uid)
                except: pass
                if ci is None:
                    try:
                        z  = round(float(frame.PlanePositionSequence[0].ImagePositionPatient[2]), 2)
                        ci = z_to_idx.get(z) or nz_fn(z)
                    except: pass
                if ci is not None:
                    fd = px[fi]
                    if fd.shape != (ny, nx):
                        from scipy.ndimage import zoom as ndzoom
                        fd = ndzoom(fd, (ny/fd.shape[0], nx/fd.shape[1]), order=0)
                    seg_masks[seg_num][ci] = np.maximum(seg_masks[seg_num][ci], fd.astype(np.uint8))
        else:
            m = np.zeros(shape, dtype=np.uint8)
            if hasattr(ds, "ReferencedSeriesSequence"):
                try:
                    refs = ds.ReferencedSeriesSequence[0].ReferencedInstanceSequence
                    for i, ref in enumerate(refs):
                        if i >= nf: break
                        ci = uid_to_idx.get(str(ref.ReferencedSOPInstanceUID))
                        if ci is not None:
                            m[ci] = np.maximum(m[ci], px[i].astype(np.uint8))
                except: pass
            if not m.any():
                for i in range(min(nf, nz)):
                    m[i] = np.maximum(m[i], px[i].astype(np.uint8))
            seg_masks[1] = m
        return seg_masks

    per_segment = defaultdict(list)
    for sp in seg_files:
        try:
            ds = pydicom.dcmread(sp)
            for seg_num, m in align_one_seg(ds).items():
                if m.any(): per_segment[seg_num].append(m)
        except: continue

    if not per_segment:
        return np.zeros(shape, dtype=np.uint8)

    segment_masks = {}
    for seg_num, masks in per_segment.items():
        n    = len(masks)
        vote = np.sum(masks, axis=0)
        segment_masks[seg_num] = (vote > n / 2).astype(np.uint8)

    biggest = max(segment_masks, key=lambda k: segment_masks[k].sum())
    mask    = segment_masks[biggest]
    print(f"[SEG] {len(per_segment)} segment(s) | retenu #{biggest} | {mask.sum():,} vox")
    return mask


# ══════════════════════════════════════════════════════════════
#  SEGMENTATION PULMONAIRE
# ══════════════════════════════════════════════════════════════

def segment_lungs(volume, prog_cb=None):
    nz, ny, nx = volume.shape
    air_vol    = (volume < -320).astype(np.uint8)
    lung_raw   = np.zeros((nz, ny, nx), dtype=np.uint8)

    for i in range(nz):
        sl = air_vol[i]
        lbl2, n2 = label(sl)
        if n2 == 0: continue
        border = set(lbl2[0,:]) | set(lbl2[-1,:]) | set(lbl2[:,0]) | set(lbl2[:,-1])
        border.discard(0)
        internal = np.zeros((ny, nx), dtype=bool)
        for lid in range(1, n2+1):
            if lid not in border: internal |= (lbl2 == lid)
        lung_raw[i] = binary_fill_holes(internal).astype(np.uint8)
        if prog_cb and i % 30 == 0: prog_cb(i, nz)

    lbl3, n3 = label(lung_raw)
    if n3 == 0: return lung_raw, np.zeros_like(lung_raw)
    sizes = np.bincount(lbl3.ravel()); sizes[0] = 0
    top2  = np.argsort(sizes)[-min(2, n3):]
    lung  = np.isin(lbl3, top2).astype(np.uint8)
    lung  = binary_closing(lung, structure=np.ones((7, 11, 11))).astype(np.uint8)
    for i in range(nz):
        lung[i] = binary_fill_holes(lung[i]).astype(np.uint8)

    # Vaisseaux — traitement par poumon séparé
    lbl_lung, n_lung = label(lung)
    sizes_lung = np.bincount(lbl_lung.ravel()); sizes_lung[0] = 0
    lung_ids   = np.argsort(sizes_lung)[-min(2, n_lung):]
    vessels    = np.zeros_like(lung)
    for lid in lung_ids:
        one_lung  = (lbl_lung == lid)
        dense_one = ((volume > -500) & (volume < 300) & one_lung).astype(np.uint8)
        lbl_v, _  = label(dense_one)
        vsizes    = np.bincount(lbl_v.ravel()); vsizes[0] = 0
        lung_vol  = int(one_lung.sum())
        keep_v    = np.where((vsizes >= 30) & (vsizes <= lung_vol * 0.15))[0]
        vessels  |= np.isin(lbl_v, keep_v).astype(np.uint8)

    print(f"[SEG] Poumon : {lung.sum():,} vox | Vaisseaux : {vessels.sum():,} vox")
    return lung, vessels


# ══════════════════════════════════════════════════════════════
#  SEGMENTATION CORPS ENTIER
# ══════════════════════════════════════════════════════════════

def segment_body(volume):
    """
    Segmente la silhouette complète du corps humain.
    Principe : tout ce qui n'est pas de l'air ambiant extérieur au patient.
    L'air extérieur = régions < -300 HU connectées au bord de l'image.
    """
    nz, ny, nx = volume.shape
    air = (volume < -300).astype(np.uint8)
    body = np.zeros((nz, ny, nx), dtype=np.uint8)

    for i in range(nz):
        sl = air[i]
        lbl2, n2 = label(sl)
        if n2 == 0:
            # Pas d'air du tout : toute la coupe = corps
            body[i] = 1
            continue
        # Régions d'air connectées au bord de l'image = air extérieur
        border_ids = set(lbl2[0, :]) | set(lbl2[-1, :]) | set(lbl2[:, 0]) | set(lbl2[:, -1])
        border_ids.discard(0)
        external_air = np.isin(lbl2, list(border_ids))
        # Corps = tout ce qui n'est PAS de l'air extérieur
        body[i] = (~external_air).astype(np.uint8)
        body[i] = binary_fill_holes(body[i]).astype(np.uint8)

    # Fermeture morphologique pour lisser et boucher les trous
    body = binary_closing(body, structure=np.ones((5, 9, 9))).astype(np.uint8)

    # Garder uniquement le plus grand composant connexe (= le corps)
    lbl3, _ = label(body)
    if lbl3.max() == 0:
        return body
    sizes = np.bincount(lbl3.ravel()); sizes[0] = 0
    body = (lbl3 == sizes.argmax()).astype(np.uint8)

    print(f"[BODY] Corps : {body.sum():,} vox")
    return body


# ══════════════════════════════════════════════════════════════
#  SEGMENTATION OS
# ══════════════════════════════════════════════════════════════
def segment_bones(volume, body_mask=None):
    """
    Segmentation os améliorée — élimine les artefacts, la table scanner,
    la peau et les fragments non-osseux.
    """
    from scipy.ndimage import binary_opening, binary_closing, binary_dilation, label

    nz, ny, nx = volume.shape

    # ── 1. Seuil HU strict (vrai os cortical > 350, évite la peau/graisse)
    bone_raw = (volume > 350).astype(np.uint8)

    # ── 2. Restriction au corps (élimine la table du scanner et l'air)
    if body_mask is not None:
        # On érode légèrement le masque corps pour exclure la peau surface
        body_eroded = binary_erosion(body_mask, iterations=4).astype(np.uint8)
        bone_raw = (bone_raw & body_eroded).astype(np.uint8)

    # ── 3. Ouverture morphologique agressive (supprime les artefacts fins)
    #    structure 3x3x3 iterations=2 = élimine tout < ~3mm
    bone_raw = binary_opening(
        bone_raw, structure=np.ones((3, 3, 3)), iterations=2
    ).astype(np.uint8)

    # ── 4. Éliminer la table du scanner
    #    La table est typiquement dans les 15% inférieurs du volume en Y
    #    et forme une ligne horizontale très plate
    table_zone = int(nz * 0.88)  # 88% vers le bas = zone table
    bone_raw[table_zone:, :, :] = 0  # coupes du bas = table scanner

    # ── 5. Éliminer les composants trop petits ET trop grands
    #    Trop petit = artefact | Trop grand = probablement pas un os
    lbl, n = label(bone_raw)
    if n == 0:
        return bone_raw
    sizes = np.bincount(lbl.ravel()); sizes[0] = 0
    total_vox = bone_raw.size

    # Seuils : garde entre 800 vox (os réel min) et 8% du volume total
    keep = np.where(
        (sizes >= 800) & (sizes <= total_vox * 0.08)
    )[0]
    bone_clean = np.isin(lbl, keep).astype(np.uint8)

    # ── 6. Fermeture pour reconnecter les fragments osseux proches (côtes)
    bone_clean = binary_closing(
        bone_clean, structure=np.ones((3, 5, 5))
    ).astype(np.uint8)

    # ── 7. Deuxième passe : supprimer les composants encore isolés < 1500 vox
    lbl2, n2 = label(bone_clean)
    if n2 > 0:
        sizes2 = np.bincount(lbl2.ravel()); sizes2[0] = 0
        keep2 = np.where(sizes2 >= 1500)[0]
        bone_clean = np.isin(lbl2, keep2).astype(np.uint8)

    print(f"[BONE] Os : {bone_clean.sum():,} vox | {len(keep2) if n2 > 0 else 0} composants")
    return bone_clean
# ══════════════════════════════════════════════════════════════
#  MAILLAGE PyVista → JSON Three.js
# ══════════════════════════════════════════════════════════════

def keep_largest_component(mask):
    lbl, n = label(mask)
    if n == 0: return mask
    sizes = np.bincount(lbl.ravel()); sizes[0] = 0
    return (lbl == sizes.argmax()).astype(np.uint8)


def mask_to_mesh_json(binary_mask, spacing, sigma=2.0, step=2,
                      smooth_iter=50, target_faces=20_000, keep_largest=True):
    if keep_largest:
        binary_mask = keep_largest_component(binary_mask)
        if binary_mask.sum() > 100:
            binary_mask = binary_erosion(binary_mask, iterations=1).astype(np.uint8)
            binary_mask = keep_largest_component(binary_mask)
    sm = gaussian_filter(binary_mask.astype(np.float32), sigma=sigma)
    if sm.max() < 0.4: return None
    try:
        verts, faces, _, _ = measure.marching_cubes(
            sm, level=0.5, spacing=spacing, step_size=step, allow_degenerate=False)
    except Exception as e:
        print(f"[WARN] marching_cubes: {e}"); return None
    if len(faces) == 0: return None

    # Construire PolyData PyVista
    n_f      = len(faces)
    pv_faces = np.hstack([np.full((n_f, 1), 3, dtype=np.int32),
                          faces.astype(np.int32)]).flatten()
    mesh = pv.PolyData(verts, pv_faces)

    # Décimation PyVista AVANT le lissage (préserve la topologie)
    current = mesh.n_cells
    if current > target_faces:
        ratio = 1.0 - (target_faces / current)
        ratio = min(ratio, 0.99)
        mesh  = mesh.decimate(ratio)
        print(f"[MESH] Decimation {current:,} -> {mesh.n_cells:,} faces")

    # Lissage Laplacien
    if smooth_iter > 0:
        mesh = mesh.smooth(n_iter=smooth_iter, relaxation_factor=0.1)

    mesh.compute_normals(inplace=True)

    pts       = np.array(mesh.points, dtype=np.float32)
    nrms      = np.array(mesh.point_data["Normals"], dtype=np.float32)
    raw_faces = np.array(mesh.faces).reshape(-1, 4)[:, 1:]

    print(f"[MESH] Final : {len(pts):,} pts | {len(raw_faces):,} faces")
    return {
        "positions": pts,
        "normals":   nrms,
        "indices":   raw_faces.astype(np.int32),
    }


def tumor_stats_dict(volume, mask, spacing):
    sz, sy, sx = spacing
    nv = int(mask.sum())
    if nv == 0: return {}
    vv = sz * sy * sx
    hu = volume[mask.astype(bool)]
    z_idx = np.where(mask.any(axis=(1,2)))[0]
    return {
        "volume_cm3":  round(nv*vv/1000, 2),
        "diameter_mm": round(2*((3*nv*vv)/(4*np.pi))**(1/3), 1),
        "hu_mean":     round(float(hu.mean()), 1),
        "hu_min":      round(float(hu.min()), 1),
        "hu_max":      round(float(hu.max()), 1),
        "z_first":     int(z_idx[0]),
        "z_last":      int(z_idx[-1]),
        "n_slices":    len(z_idx),
    }


# ══════════════════════════════════════════════════════════════
#  PIPELINE
# ══════════════════════════════════════════════════════════════

def process_patient(folder):
    global _cache, _progress
    def prog(pct, msg):
        global _progress
        _progress = {"pct": pct, "msg": msg, "done": False, "error": None}
        print(f"[{pct:3d}%] {msg}")
    try:
        prog(5,  "Scan des fichiers DICOM…")
        ct_files, seg_files = find_ct_and_seg(folder)
        if not ct_files: raise ValueError("Aucun fichier CT trouve.")

        prog(15, f"Chargement {len(ct_files)} coupes CT…")
        volume, spacing, sop_uids, slices = load_ct(ct_files)

        prog(28, "Chargement segmentations SEG…")
        mask  = load_seg_mask(seg_files, sop_uids, volume.shape, slices) \
                if seg_files else np.zeros(volume.shape, dtype=np.uint8)
        stats = tumor_stats_dict(volume, mask, spacing)

        prog(38, "Segmentation corps entier…")
        body_mask = segment_body(volume)

        prog(45, "Segmentation pulmonaire…")
        def seg_cb(i, total):
            _progress["pct"] = 45 + int(15 * i / total)
            _progress["msg"] = f"Segmentation coupe {i}/{total}…"
        lung_mask, vessels_mask = segment_lungs(volume, prog_cb=seg_cb)

        prog(58, "Segmentation os…")
        bone_mask = segment_bones(volume, body_mask)

        prog(62, "Maillage corps entier (PyVista)…")
        body_raw = mask_to_mesh_json(body_mask, spacing,
                                     sigma=3.5, step=3, smooth_iter=100,
                                     target_faces=80_000)

        prog(72, "Maillage poumons (PyVista)…")
        lung_raw = mask_to_mesh_json(lung_mask, spacing,
                                     sigma=2.5, step=2, smooth_iter=60,
                                     target_faces=60_000)

        prog(80, "Maillage vaisseaux (PyVista)…")
        lbl_v2, nv2 = label(vessels_mask)
        if nv2 > 0:
            sv2 = np.bincount(lbl_v2.ravel()); sv2[0] = 0
            top_v = [i for i in np.argsort(sv2)[-min(80,nv2):] if sv2[i] >= 200]
            vessels_mask = np.isin(lbl_v2, top_v).astype(np.uint8)
        vessels_raw = mask_to_mesh_json(vessels_mask, spacing,
                                        sigma=1.0, step=1, smooth_iter=25,
                                        target_faces=40_000, keep_largest=False)

        prog(88, "Maillage os (PyVista)…")
        bone_raw = mask_to_mesh_json(bone_mask, spacing,
                                     sigma=1.2, step=2, smooth_iter=40,
                                     target_faces=100_000, keep_largest=False)

        prog(93, "Maillage nodule (PyVista)…")
        nodule_raw = None
        if mask.any():
            nodule_raw = mask_to_mesh_json(mask, spacing,
                                           sigma=1.0, step=1, smooth_iter=40,
                                           target_faces=12_000)

        # Centrage global sur le corps entier (ou poumon si pas de corps)
        ref_raw = body_raw if body_raw is not None else lung_raw
        if ref_raw is not None:
            pts_ref = ref_raw["positions"]
            global_center = (pts_ref.max(axis=0) + pts_ref.min(axis=0)) / 2.0
        else:
            global_center = np.zeros(3, dtype=np.float32)

        def finalize(raw):
            if raw is None: return None
            pts  = raw["positions"] - global_center
            nrms = raw["normals"]
            # skimage: (slice, row, col) -> Three.js: (X=col, Y=slice, Z=row)
            pts_r  = pts[:,  [2, 0, 1]].copy()
            nrms_r = nrms[:, [2, 0, 1]].copy()
            return {
                "positions": pts_r.flatten().tolist(),
                "normals":   nrms_r.flatten().tolist(),
                "indices":   raw["indices"].flatten().tolist(),
                "center":    global_center.tolist(),
            }

        body_json    = finalize(body_raw)
        lung_json    = finalize(lung_raw)
        vessels_json = finalize(vessels_raw)
        bone_json    = finalize(bone_raw)
        nodule_json  = finalize(nodule_raw)

        prog(96, "Serialisation…")
        _cache = {
            "body_mesh":    body_json,
            "lung_mesh":    lung_json,
            "vessels_mesh": vessels_json,
            "bone_mesh":    bone_json,
            "nodule_mesh":  nodule_json,
            "stats":        stats,
            "n_ct":         len(ct_files),
            "n_seg":        len(seg_files),
            "volume":       volume,
            "seg_mask":     mask,
        }
        _progress = {"pct": 100, "msg": "Pret !", "done": True, "error": None}
        print("[OK] Traitement termine.")
    except Exception as e:
        import traceback; traceback.print_exc()
        _progress = {"pct": 0, "msg": str(e), "done": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/load", methods=["POST"])
def api_load():
    data   = request.get_json()
    folder = data.get("folder", "").strip()
    if not os.path.isdir(folder):
        return jsonify({"error": f"Dossier introuvable : {folder}"}), 400
    threading.Thread(target=process_patient, args=(folder,), daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/progress")
def api_progress():
    return jsonify(_progress)

@app.route("/api/meshes")
def api_meshes():
    if not _cache:
        return jsonify({"error": "Aucun volume charge"}), 404
    return jsonify({
        "body_mesh":    _cache["body_mesh"],
        "lung_mesh":    _cache["lung_mesh"],
        "vessels_mesh": _cache["vessels_mesh"],
        "bone_mesh":    _cache["bone_mesh"],
        "nodule_mesh":  _cache["nodule_mesh"],
        "stats":        _cache["stats"],
        "n_ct":         _cache["n_ct"],
        "n_seg":        _cache["n_seg"],
    })

@app.route("/api/slice")
def api_slice():
    import base64, io
    if not _cache or "volume" not in _cache:
        return jsonify({"error": "Pas de volume"}), 404
    try:
        from PIL import Image
    except ImportError:
        return jsonify({"error": "Pillow non installe : pip install Pillow"}), 500

    axis = request.args.get("axis", "axial")
    idx  = int(request.args.get("idx", 0))
    vol  = _cache["volume"]
    seg  = _cache["seg_mask"]
    nz, ny, nx = vol.shape

    if axis == "axial":
        idx = max(0, min(idx, nz - 1))
        sl  = vol[idx];       sm = seg[idx]
        total = nz
    elif axis == "coronal":
        idx = max(0, min(idx, ny - 1))
        sl  = vol[:, idx, :]; sm = seg[:, idx, :]
        total = ny
    else:  # sagittal
        idx = max(0, min(idx, nx - 1))
        sl  = vol[:, :, idx]; sm = seg[:, :, idx]
        total = nx

    # Fenetre pulmonaire HU -1200 → +400
    sl_norm = ((np.clip(sl, -1200, 400) + 1200) / 1600 * 255).astype(np.uint8)
    rgb = np.stack([sl_norm, sl_norm, sl_norm], axis=-1)

    # Overlay nodule en rouge vif
    if sm.any():
        mask_px = sm > 0
        rgb[mask_px, 0] = 255
        rgb[mask_px, 1] = (rgb[mask_px, 1].astype(int) * 0.2).astype(np.uint8)
        rgb[mask_px, 2] = (rgb[mask_px, 2].astype(int) * 0.2).astype(np.uint8)

    # Orientation anatomique : tete en haut
    rgb = np.flipud(rgb)

    img = Image.fromarray(rgb, "RGB")
    w, h = img.size
    scale = min(512 / max(w, 1), 512 / max(h, 1))
    img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.NEAREST)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return jsonify({"img": b64, "idx": idx, "total": total})


if __name__ == "__main__":
    print("\n  PulmoCare Flask+PyVista  http://localhost:5000\n")
    app.run(debug=False, port=5000, threaded=True)
