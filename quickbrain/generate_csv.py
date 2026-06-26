"""Build a flat CSV-ready record from any combination of QuiCk-Brain module outputs.

Usage
-----
    from quickbrain.artifacts import detect_artifacts
    from quickbrain.contrast  import detect_contrast_enhancement
    from quickbrain.generate_csv import build_qc_record

    art = detect_artifacts(image, brain_mask)
    con = detect_contrast_enhancement(image, brain_mask)

    record = build_qc_record(
        image_path='T1w.nii.gz',
        patient_id='sub-001',
        artifacts=art,
        contrast=con,
    )
"""

from datetime import datetime
from pathlib import Path

from quickbrain.schema import ALL_COLUMNS, ARTIFACT_CLASSES


def build_qc_record(
    image_path: str,
    patient_id: str = None,
    artifacts: dict = None,
    contrast: dict = None,
    coreg: dict = None,
    fov: dict = None,
    meta: dict = None,
    timestamp: str = None,
) -> dict:
    """Flatten module outputs into a flat dict matching the canonical CSV schema.

    Parameters
    ----------
    image_path  : path to the NIfTI scan — stored as scan identifier.
    patient_id  : optional label; defaults to the NIfTI filename stem.
    artifacts   : dict returned by detect_artifacts(), or None.
    contrast    : dict returned by detect_contrast_enhancement(), or None.
    coreg       : dict returned by registration_qc(), or None.
    fov         : dict returned by check_fov(), or None.
    meta        : dict returned by metaqc.run_qc(), or None.
    timestamp   : ISO datetime string; defaults to now.

    Returns
    -------
    dict with ALL_COLUMNS keys.  Columns for modules not supplied are ''.
    """
    record = dict.fromkeys(ALL_COLUMNS, '')
    record['timestamp']  = timestamp or datetime.now().strftime('%Y-%m-%d %H:%M')
    record['scan_path']  = str(image_path)
    record['patient_id'] = patient_id or Path(image_path).stem

    if artifacts is not None:
        # Prefer scaled [0,1] severity (regression model output after calibration).
        # Falls back to raw regression scores or old softmax probabilities.
        scores = (
            artifacts.get('artifact_severity_scaled')
            or artifacts.get('artifact_severity')
            or artifacts.get('artifact_probabilities', {})
        )
        record['artifacts_quality_passed'] = artifacts.get('quality_passed', '')
        detected = artifacts.get('artifacts_detected', [])
        record['artifacts_detected'] = '|'.join(detected) if detected else ''
        # prob_clean always empty in regression mode; kept for schema compatibility.
        for cls in ARTIFACT_CLASSES:
            record[f'prob_{cls}'] = scores.get(cls, '')
        iqms = artifacts.get('iqms', {})
        record['iqm_motion_blur_score'] = iqms.get('motion_blur_score', '')
        record['iqm_snr']               = iqms.get('snr', '')

    if contrast is not None:
        record['contrast_enhanced']              = contrast.get('enhanced', '')
        record['contrast_vessel_ratio']          = contrast.get('vessel_ratio', '')
        record['contrast_bright_voxel_fraction'] = contrast.get('bright_voxel_fraction', '')

    if coreg is not None:
        record['coreg_flag'] = coreg.get('flag', '')
        record['coreg_ssim'] = coreg.get('ssim', '')
        record['coreg_ncc']  = coreg.get('ncc', '')
        passed = coreg.get('passed', {})
        record['coreg_ssim_passed'] = passed.get('ssim', '')
        record['coreg_ncc_passed']  = passed.get('ncc', '')

    if fov is not None:
        record['fov_overall'] = fov.get('Overall', '')
        def _join(v):
            return '|'.join(v) if isinstance(v, list) else str(v)
        record['fov_check1'] = _join(fov.get('Check 1 (scan edge proximity)', []))
        record['fov_check2'] = _join(fov.get('Check 2 (margin proximity)', []))
        record['fov_check3'] = _join(fov.get('Check 3 (distance check)', []))

    if meta is not None:
        record['metaqc_status']   = meta.get('status', '')
        reasons = meta.get('reasons', [])
        record['metaqc_reasons']  = '|'.join(reasons) if reasons else ''
        features = meta.get('features', {})
        record['metaqc_foreground_fraction'] = features.get('foreground_fraction', '')
        record['metaqc_intensity_mean']      = features.get('intensity_mean', '')
        record['metaqc_intensity_std']       = features.get('intensity_std', '')
        record['metaqc_centroid_offset_mm']  = features.get('centroid_offset_mm', '')
        meta_qc = meta.get('metadata_qc', {})
        record['metaqc_metadata_status'] = meta_qc.get('status', '')

        # Image geometry is already extracted by metaqc.extract_metadata() — reuse
        # it rather than reloading the NIfTI.
        hdr_meta = meta_qc.get('metadata', {})
        shape  = hdr_meta.get('shape') or []
        zooms  = hdr_meta.get('voxel_spacing') or []
        record['img_dim_x']       = int(shape[0]) if len(shape) > 0 else ''
        record['img_dim_y']       = int(shape[1]) if len(shape) > 1 else ''
        record['img_dim_z']       = int(shape[2]) if len(shape) > 2 else ''
        record['img_vox_x']       = zooms[0] if len(zooms) > 0 else ''
        record['img_vox_y']       = zooms[1] if len(zooms) > 1 else ''
        record['img_vox_z']       = zooms[2] if len(zooms) > 2 else ''
        record['img_orientation'] = hdr_meta.get('orientation', '')
    else:
        # Fallback: header-only read (no voxel data loaded) when metaqc not run.
        try:
            import nibabel as nib
            _img  = nib.load(str(image_path))
            _dims = _img.header.get_data_shape()
            _zooms = _img.header.get_zooms()
            record['img_dim_x']       = int(_dims[0]) if len(_dims) > 0 else ''
            record['img_dim_y']       = int(_dims[1]) if len(_dims) > 1 else ''
            record['img_dim_z']       = int(_dims[2]) if len(_dims) > 2 else ''
            record['img_vox_x']       = round(float(_zooms[0]), 4) if len(_zooms) > 0 else ''
            record['img_vox_y']       = round(float(_zooms[1]), 4) if len(_zooms) > 1 else ''
            record['img_vox_z']       = round(float(_zooms[2]), 4) if len(_zooms) > 2 else ''
            record['img_orientation'] = ''.join(nib.aff2axcodes(_img.affine))
        except Exception:
            pass

    return record