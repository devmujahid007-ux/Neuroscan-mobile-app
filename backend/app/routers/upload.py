import io
import json
import os
import re
import shutil
import types
import zipfile
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload
from app.security.jwt import role_required
from app.database.db import SessionLocal
from app.models.medical import MRIScan, ScanStatus, Diagnosis, Report
from app.models.user import User
from app.schemas.medical import MRIScanOut
from app.ml.monai_preprocess import _build_file_map, _validate_file_map
from pydantic import BaseModel

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
os.makedirs(DATA_DIR, exist_ok=True)
UPLOAD_DIR = DATA_DIR
SCANS_UPLOAD_DIR = os.path.join(DATA_DIR, "scans")
os.makedirs(SCANS_UPLOAD_DIR, exist_ok=True)
for _kind in ("tumor", "alzheimer"):
    os.makedirs(os.path.join(SCANS_UPLOAD_DIR, _kind), exist_ok=True)
MRI_MODALITY_ORDER = ("t1c", "t1n", "t2f", "t2w")
ALLOWED_MRI_EXTS = {".dcm", ".nii", ".nii.gz"}
MAX_ZIP_BYTES = int(os.getenv("MRI_ZIP_MAX_BYTES", str(500 * 1024 * 1024)))
ALLOWED_ALZ_IMAGE_EXT = {".png", ".jpg", ".jpeg"}
MAX_ALZ_IMAGE_BYTES = int(os.getenv("ALZ_IMAGE_MAX_BYTES", str(20 * 1024 * 1024)))

router = APIRouter(prefix="/mri", tags=["MRI"])


class SendToDoctorPayload(BaseModel):
    doctor_id: int

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

WORKFLOW_STATUSES = (ScanStatus.sent, ScanStatus.analyzed, ScanStatus.reported)


def _public_uploads_url(abs_path: str) -> str | None:
    """Expose a file under ``UPLOAD_DIR`` as ``/uploads/...`` (for patient/doctor dashboards)."""
    if not abs_path:
        return None
    try:
        candidate = os.path.abspath(abs_path)
        if not os.path.isfile(candidate):
            return None
        rel = os.path.relpath(candidate, UPLOAD_DIR).replace("\\", "/")
        if rel.startswith(".."):
            return None
        return f"/uploads/{rel}"
    except Exception:
        return None


def _get_upload_ext(upload: UploadFile) -> str:
    filename = (upload.filename or "").lower()
    if filename.endswith(".nii.gz"):
        return ".nii.gz"
    return os.path.splitext(filename)[1]


def _read_and_validate_upload(upload: UploadFile, modality: str) -> tuple[bytes, str]:
    ext = _get_upload_ext(upload)
    if ext not in ALLOWED_MRI_EXTS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type for '{modality}': {upload.filename}. "
                "Allowed: .dcm, .nii, .nii.gz"
            ),
        )

    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail=f"Uploaded file for '{modality}' is empty")

    if ext == ".dcm" and (len(data) < 132 or data[128:132] != b"DICM"):
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file for '{modality}' does not appear to be a valid DICOM file",
        )

    return data, ext


def _validate_all_modalities_present(files_by_modality: dict[str, Optional[UploadFile]]) -> None:
    if any(files_by_modality.get(modality) is None for modality in MRI_MODALITY_ORDER):
        raise HTTPException(
            status_code=400,
            detail="All 4 MRI modalities required: t1c, t1n, t2f, t2w",
        )


def _upload_nonempty(f: Optional[UploadFile]) -> bool:
    """True if multipart part has a filename or a non-empty body (Flutter Web may omit filename)."""
    if f is None:
        return False
    if (f.filename or "").strip():
        return True
    try:
        fp = f.file
        pos = fp.tell()
    except Exception:
        return False
    try:
        fp.seek(0, os.SEEK_END)
        size = fp.tell()
        fp.seek(0)
        return size > 0
    except Exception:
        try:
            fp.seek(pos)
        except Exception:
            pass
        return False


def _use_zip_upload(mri_zip: Optional[UploadFile]) -> bool:
    """Detect patient ZIP: ``.zip`` filename or ZIP local file header (``PK``) after rewind."""
    if mri_zip is None or not _upload_nonempty(mri_zip):
        return False
    if (mri_zip.filename or "").strip().lower().endswith(".zip"):
        try:
            mri_zip.file.seek(0)
        except Exception:
            pass
        return True
    try:
        fp = mri_zip.file
        fp.seek(0)
        head = fp.read(4)
        fp.seek(0)
        return len(head) >= 2 and head[0] == 0x50 and head[1] == 0x4B
    except Exception:
        return False


def _scan_storage_kind(scan_kind: str | None) -> str:
    return "alzheimer" if (scan_kind or "").lower() == "alzheimer" else "tumor"


def _scan_storage_dir(scan_id: int, scan_kind: str | None) -> str:
    return os.path.join(SCANS_UPLOAD_DIR, _scan_storage_kind(scan_kind), str(scan_id))


def _unlink_if_under_data(path: str | None) -> None:
    """Remove a file or directory only if it lives under ``DATA_DIR`` (safety)."""
    if not path or not str(path).strip():
        return
    try:
        p = os.path.abspath(str(path).strip())
        root = os.path.abspath(DATA_DIR)
        if p != root and not p.startswith(root + os.sep):
            return
        if os.path.isfile(p):
            os.remove(p)
        elif os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
    except OSError:
        pass


def _abs_path_from_public_ref(ref: str | None) -> str | None:
    """Map ``/uploads/...`` or ``/outputs/...`` (or absolute paths under data) to a filesystem path."""
    if not ref or not str(ref).strip():
        return None
    raw = str(ref).strip()
    if raw.startswith("http://") or raw.startswith("https://"):
        raw = (urlparse(raw).path or "").strip() or raw
    if os.path.isabs(raw):
        p = os.path.abspath(raw)
        root = os.path.abspath(DATA_DIR)
        if p == root or p.startswith(root + os.sep):
            return p
        return None
    parsed = raw if raw.startswith("/") else f"/{raw}"
    if parsed.startswith("/uploads/"):
        rel = parsed[len("/uploads/") :].lstrip("/").replace("/", os.sep)
        return os.path.abspath(os.path.join(DATA_DIR, rel))
    if parsed.startswith("/outputs/"):
        rel = parsed[len("/outputs/") :].lstrip("/").replace("/", os.sep)
        return os.path.abspath(os.path.join(DATA_DIR, "results", "tumor", rel))
    return None


def _unlink_public_ref(ref: str | None) -> None:
    p = _abs_path_from_public_ref(ref)
    if p:
        _unlink_if_under_data(p)


def _unlink_diagnosis_result_assets(dx: Diagnosis) -> None:
    """Remove on-disk result images referenced by a diagnosis (paths or /uploads/... URLs)."""
    _unlink_public_ref(dx.result_image_path)
    for text in (dx.result_payload, dx.model_meta):
        if not text:
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        for k in ("overlay_image", "reference_mri_png", "input_image"):
            v = obj.get(k)
            if isinstance(v, str):
                _unlink_public_ref(v)


def _save_modalities_to_scan_dir(scan_id: int, files_by_modality: dict[str, UploadFile], scan_kind: str = "mri") -> str:
    scan_dir = _scan_storage_dir(scan_id, scan_kind)
    if os.path.isdir(scan_dir):
        shutil.rmtree(scan_dir)
    os.makedirs(scan_dir, exist_ok=True)

    for modality in MRI_MODALITY_ORDER:
        upload = files_by_modality[modality]
        data, ext = _read_and_validate_upload(upload, modality)
        file_path = os.path.join(scan_dir, f"{modality}{ext}")
        with open(file_path, "wb") as out:
            out.write(data)

    return scan_dir


def _list_nifti_paths(root: str) -> list[str]:
    paths: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            low = fn.lower()
            if low.endswith(".nii.gz") or low.endswith(".nii"):
                paths.append(os.path.join(dirpath, fn))
    return paths


def _nifti_stem_from_basename(filename: str) -> str:
    low = filename.lower()
    if low.endswith(".nii.gz"):
        return low[: -len(".nii.gz")]
    if low.endswith(".nii"):
        return low[: -len(".nii")]
    return low


def _modality_token_in_stem(stem: str, modality: str) -> bool:
    """True if ``modality`` appears as its own token (e.g. avoid matching *t1c* inside *t1ce*)."""
    return re.search(rf"(?<![a-z0-9]){re.escape(modality)}(?![a-z0-9])", stem) is not None


def _resolve_modalities_from_extracted_zip(root: str) -> dict[str, str]:
    """
    Map up to four NIfTI volumes to BraTS-style keys for MONAI.

    - If the archive contains **exactly four** NIfTI files, they are accepted in sorted path order
      (no required filenames).
    - If there are more than four, we try exact names, then stems, then token matches (t1c/t1n/t2f/t2w).
    """
    niftis = sorted(set(_list_nifti_paths(root)))
    if len(niftis) < 4:
        raise ValueError(
            f"This ZIP needs at least four NIfTI volumes (.nii or .nii.gz). Found {len(niftis)}."
        )

    if len(niftis) == 4:
        return dict(zip(MRI_MODALITY_ORDER, niftis))

    found: dict[str, str] = {}
    used: set[str] = set()

    def take(path: str, modality: str) -> None:
        if modality not in found:
            found[modality] = path
            used.add(path)

    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            low = fn.lower()
            if not (low.endswith(".nii.gz") or low.endswith(".nii")):
                continue
            path = os.path.join(dirpath, fn)
            if path in used:
                continue
            for m in MRI_MODALITY_ORDER:
                if m in found:
                    continue
                if low == f"{m}.nii.gz" or low == f"{m}.nii":
                    take(path, m)
                    break

    for path in niftis:
        if path in used:
            continue
        stem = _nifti_stem_from_basename(os.path.basename(path))
        for m in MRI_MODALITY_ORDER:
            if m in found:
                continue
            if stem == m:
                take(path, m)
                break

    for path in niftis:
        if path in used:
            continue
        stem = _nifti_stem_from_basename(os.path.basename(path))
        for m in MRI_MODALITY_ORDER:
            if m in found:
                continue
            if _modality_token_in_stem(stem, m):
                take(path, m)
                break

    if len(found) == 4:
        return found

    raise ValueError(
        f"This ZIP has {len(niftis)} NIfTI files. Use a ZIP with exactly four volumes, or name files so "
        "each modality (t1c, t1n, t2f, t2w) can be told apart in the filename."
    )


def _save_modalities_from_zip(scan_id: int, upload: UploadFile, scan_kind: str = "mri") -> str:
    """Extract ZIP, resolve four NIfTI volumes, copy into flat scan folder for MONAI."""
    kind_root = os.path.join(SCANS_UPLOAD_DIR, _scan_storage_kind(scan_kind))
    scan_dir = _scan_storage_dir(scan_id, scan_kind)
    if os.path.isdir(scan_dir):
        shutil.rmtree(scan_dir)
    extract_root = os.path.join(scan_dir, "__extract__")
    os.makedirs(extract_root, exist_ok=True)
    try:
        raw = upload.file.read()
        if not raw:
            if os.path.isdir(scan_dir):
                shutil.rmtree(scan_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail="ZIP file is empty")
        if len(raw) > MAX_ZIP_BYTES:
            if os.path.isdir(scan_dir):
                shutil.rmtree(scan_dir, ignore_errors=True)
            raise HTTPException(
                status_code=400,
                detail=f"ZIP too large (max {MAX_ZIP_BYTES // (1024 * 1024)} MB)",
            )
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = info.filename.replace("\\", "/").strip("/")
                if not name or ".." in name.split("/"):
                    raise HTTPException(status_code=400, detail="ZIP contains invalid paths")
                dest_path = os.path.join(extract_root, name)
                abs_extract = os.path.abspath(extract_root)
                abs_dest = os.path.abspath(dest_path)
                if not abs_dest.startswith(abs_extract + os.sep) and abs_dest != abs_extract:
                    raise HTTPException(status_code=400, detail="ZIP path escapes extraction folder")
                os.makedirs(os.path.dirname(abs_dest), exist_ok=True)
                with zf.open(info) as src, open(abs_dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)

        try:
            found = _resolve_modalities_from_extracted_zip(extract_root)
        except ValueError as e:
            if os.path.isdir(scan_dir):
                shutil.rmtree(scan_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(e)) from e

        # Copy into a sibling staging dir first — ``found`` paths live under ``scan_dir/__extract__``,
        # so we must not ``rmtree(scan_dir)`` before copying (that caused 500s / broken uploads).
        staging = os.path.join(kind_root, f"{scan_id}.staging")
        if os.path.isdir(staging):
            shutil.rmtree(staging, ignore_errors=True)
        os.makedirs(staging, exist_ok=True)
        try:
            for m in MRI_MODALITY_ORDER:
                src = found[m]
                ext = ".nii.gz" if src.lower().endswith(".nii.gz") else ".nii"
                shutil.copy2(src, os.path.join(staging, f"{m}{ext}"))
        except OSError:
            shutil.rmtree(staging, ignore_errors=True)
            if os.path.isdir(scan_dir):
                shutil.rmtree(scan_dir, ignore_errors=True)
            raise

        if os.path.isdir(scan_dir):
            shutil.rmtree(scan_dir)
        shutil.move(staging, scan_dir)

        try:
            fm = _build_file_map(scan_dir)
            _validate_file_map(fm)
        except FileNotFoundError as e:
            if os.path.isdir(scan_dir):
                shutil.rmtree(scan_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(e)) from e
        return scan_dir
    except zipfile.BadZipFile as e:
        if os.path.isdir(scan_dir):
            shutil.rmtree(scan_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Invalid ZIP file: {e}") from e
    finally:
        staging_left = os.path.join(kind_root, f"{scan_id}.staging")
        if os.path.isdir(staging_left):
            shutil.rmtree(staging_left, ignore_errors=True)
        extract_gone = os.path.join(_scan_storage_dir(scan_id, scan_kind), "__extract__")
        if os.path.isdir(extract_gone):
            shutil.rmtree(extract_gone, ignore_errors=True)


@router.post("/upload", response_model=MRIScanOut)
def upload_mri(
    t1c: Optional[UploadFile] = File(default=None),
    t1n: Optional[UploadFile] = File(default=None),
    t2f: Optional[UploadFile] = File(default=None),
    t2w: Optional[UploadFile] = File(default=None),
    mri_zip: Optional[UploadFile] = File(default=None),
    doctor_id: Optional[int] = Form(default=None),
    db: Session = Depends(get_db),
    current = Depends(role_required("patient", "doctor", "admin")),
):
    """Upload MRI files for a single scan.

    **Patients** must upload a single ``mri_zip`` with at least four NIfTI volumes (``.nii`` / ``.nii.gz``).
    If the archive contains exactly four such files, names do not matter; otherwise filenames should
    distinguish the four BraTS modalities (t1c, t1n, t2f, t2w).

    Patients: must pass ``doctor_id`` so the scan is assigned immediately (status ``sent``) and
    appears on that doctor's dashboard. Uploads without a doctor are rejected to avoid scans stuck
    in ``pending`` that doctors never see.

    Doctors: scan is stored as a doctor-owned upload (not shown on the patient-doctor request queue).
    Doctors must use four separate modality files (ZIP is patient-only).
    """
    role = (current.role or "").lower()
    files_by_modality = {
        "t1c": t1c,
        "t1n": t1n,
        "t2f": t2f,
        "t2w": t2w,
    }
    use_zip = _use_zip_upload(mri_zip)
    zip_name = ((mri_zip.filename or "").strip() if mri_zip else "") or (
        "upload.zip" if use_zip else ""
    )

    if role == "patient":
        if not use_zip:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Patients must upload one ZIP file with your MRI volumes "
                    "(at least four .nii or .nii.gz files; see upload help text on the dashboard)."
                ),
            )
    elif use_zip:
        raise HTTPException(
            status_code=400,
            detail="ZIP upload is only for patient accounts; upload four modality files instead.",
        )
    else:
        _validate_all_modalities_present(files_by_modality)
    patient_id = None
    scan_doctor_id = None
    status = ScanStatus.pending
    sent_date = None

    if role == "patient":
        patient_id = current.id
        assign_id = doctor_id if doctor_id and doctor_id > 0 else None
        if assign_id is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "doctor_id is required for patient uploads. Choose which doctor should receive "
                    "this MRI; otherwise it stays pending and will not appear on any doctor dashboard."
                ),
            )
        doctor = (
            db.query(User)
            .filter(User.id == assign_id, func.lower(User.role) == "doctor")
            .first()
        )
        if not doctor:
            raise HTTPException(status_code=404, detail="Doctor not found")
        scan_doctor_id = doctor.id
        status = ScanStatus.sent
        sent_date = datetime.utcnow()
    elif role == "doctor":
        scan_doctor_id = current.id

    scan = MRIScan(
        patient_id=patient_id,
        doctor_id=scan_doctor_id,
        file_path="",
        original_filename=zip_name if use_zip else ",".join((files_by_modality[m].filename or "") for m in MRI_MODALITY_ORDER),
        upload_source="patient_zip" if use_zip else "multi_file",
        scan_kind="mri",
        status=status,
        upload_date=datetime.utcnow(),
        sent_date=sent_date,
    )
    db.add(scan)
    db.flush()
    if use_zip:
        fpath = _save_modalities_from_zip(scan.id, mri_zip, "mri")
    else:
        fpath = _save_modalities_to_scan_dir(scan.id, files_by_modality, "mri")
    scan.file_path = fpath
    db.commit()
    db.refresh(scan)
    return scan


@router.post("/upload-zip", response_model=MRIScanOut)
def upload_patient_mri_zip_only(
    mri_zip: UploadFile = File(..., description="Single MRI .zip (patient accounts only)"),
    doctor_id: int = Form(...),
    db: Session = Depends(get_db),
    current=Depends(role_required("patient")),
):
    """Patient ZIP upload without ``t1c``/``t1n``/… multipart fields (avoids stray modality validation)."""
    payload = mri_zip.file.read()
    if len(payload) < 4 or payload[0] != 0x50 or payload[1] != 0x4B:
        raise HTTPException(
            status_code=400,
            detail="Upload a valid ZIP archive (form field mri_zip).",
        )
    display_name = (mri_zip.filename or "").strip() or "upload.zip"
    if not display_name.lower().endswith(".zip"):
        display_name = f"{display_name}.zip"
    zip_upload = types.SimpleNamespace(
        filename=display_name,
        file=io.BytesIO(payload),
    )

    assign_id = doctor_id if doctor_id > 0 else None
    if assign_id is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "doctor_id is required. Choose which doctor should receive this MRI."
            ),
        )
    doctor = (
        db.query(User)
        .filter(User.id == assign_id, func.lower(User.role) == "doctor")
        .first()
    )
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")

    scan = MRIScan(
        patient_id=current.id,
        doctor_id=doctor.id,
        file_path="",
        original_filename=display_name,
        upload_source="patient_zip",
        scan_kind="mri",
        status=ScanStatus.sent,
        upload_date=datetime.utcnow(),
        sent_date=datetime.utcnow(),
    )
    db.add(scan)
    db.flush()
    fpath = _save_modalities_from_zip(scan.id, zip_upload, "mri")
    scan.file_path = fpath
    db.commit()
    db.refresh(scan)
    return scan


@router.post("/upload-alz-image", response_model=MRIScanOut)
@router.post("/upload-alzheimer-png", response_model=MRIScanOut)
def upload_alzheimer_patient_image(
    image: Optional[UploadFile] = File(default=None, description="PNG or JPEG brain MRI slice / image"),
    mri_png: Optional[UploadFile] = File(default=None, description="Mobile alias for Alzheimer image upload"),
    doctor_id: int = Form(...),
    db: Session = Depends(get_db),
    current=Depends(role_required("patient")),
):
    """
    Patient-only: upload a single image for Alzheimer detection (not the tumor ZIP pipeline).
    Stored under ``uploads/scans/<id>/`` and queued to the selected doctor (status ``sent``).
    """
    upload = image or mri_png
    if upload is None:
        raise HTTPException(
            status_code=422,
            detail="Image file is required (form field: image or mri_png).",
        )

    name = (upload.filename or "").strip().lower()
    ext = ""
    if name.endswith(".jpeg"):
        ext = ".jpeg"
    elif name.endswith(".jpg"):
        ext = ".jpg"
    elif name.endswith(".png"):
        ext = ".png"
    if ext not in ALLOWED_ALZ_IMAGE_EXT:
        raise HTTPException(
            status_code=400,
            detail="Allowed image types: .png, .jpg, .jpeg",
        )

    assign_id = doctor_id if doctor_id > 0 else None
    if assign_id is None:
        raise HTTPException(status_code=400, detail="doctor_id is required.")
    doctor = (
        db.query(User)
        .filter(User.id == assign_id, func.lower(User.role) == "doctor")
        .first()
    )
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")

    raw = upload.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Image file is empty")
    if len(raw) > MAX_ALZ_IMAGE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large (max {MAX_ALZ_IMAGE_BYTES // (1024 * 1024)} MB)",
        )

    display_name = (upload.filename or "").strip() or f"alz_upload{ext}"
    scan = MRIScan(
        patient_id=current.id,
        doctor_id=doctor.id,
        file_path="",
        original_filename=display_name,
        upload_source="patient_alz_image",
        scan_kind="alzheimer",
        status=ScanStatus.sent,
        upload_date=datetime.utcnow(),
        sent_date=datetime.utcnow(),
        upload_size_bytes=len(raw),
    )
    db.add(scan)
    db.flush()

    scan_dir = _scan_storage_dir(scan.id, "alzheimer")
    if os.path.isdir(scan_dir):
        shutil.rmtree(scan_dir, ignore_errors=True)
    os.makedirs(scan_dir, exist_ok=True)
    dest_name = f"alz_image{ext}"
    dest_abs = os.path.join(scan_dir, dest_name)
    with open(dest_abs, "wb") as out:
        out.write(raw)

    scan.file_path = dest_abs
    db.commit()
    db.refresh(scan)
    return scan


@router.post("/send-to-doctor/{scan_id}")
def send_scan_to_doctor(
    scan_id: int,
    payload: SendToDoctorPayload,
    db: Session = Depends(get_db),
    current = Depends(role_required("patient"))
):
    """Patient sends their MRI scan to a doctor for analysis."""
    scan = db.query(MRIScan).filter(MRIScan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    if scan.patient_id is None or scan.patient_id != current.id:
        raise HTTPException(
            status_code=403,
            detail="Unauthorized: You can only send your own scans (re-upload if this scan has no patient owner).",
        )

    doctor = (
        db.query(User)
        .filter(User.id == payload.doctor_id, func.lower(User.role) == "doctor")
        .first()
    )
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")

    # Update scan with doctor assignment (single source of truth for the doctor inbox)
    scan.doctor_id = doctor.id
    scan.status = ScanStatus.sent
    scan.sent_date = datetime.utcnow()
    db.commit()
    db.refresh(scan)

    return {
        "message": "Scan sent to doctor successfully",
        "scan_id": scan.id,
        "status": scan.status.value if hasattr(scan.status, "value") else scan.status,
        "doctor_id": doctor.id,
        "doctor": {
            "id": doctor.id,
            "name": doctor.name,
            "email": doctor.email,
            "phone": doctor.phone,
        },
    }


@router.get("/patient-scans")
def get_patient_scans(
    db: Session = Depends(get_db),
    current = Depends(role_required("patient"))
):
    """Get all MRI scans uploaded by the current patient."""
    scans = (
        db.query(MRIScan)
        .filter(MRIScan.patient_id == current.id)
        .order_by(MRIScan.upload_date.desc(), MRIScan.id.desc())
        .all()
    )

    out = []
    for scan in scans:
        fp = (scan.file_path or "").strip()
        filename = os.path.basename(fp) if fp else ""
        file_url = _public_uploads_url(fp) if fp else None
        sk = getattr(scan, "scan_kind", None) or "mri"
        out.append({
            "id": scan.id,
            "patient_id": scan.patient_id,
            "doctor_id": scan.doctor_id,
            "file_path": scan.file_path,
            "file_name": filename,
            "file_url": file_url,
            "scan_kind": sk,
            "status": scan.status.value if hasattr(scan.status, "value") else scan.status,
            "upload_date": scan.upload_date.isoformat() if scan.upload_date else None,
            "sent_date": scan.sent_date.isoformat() if scan.sent_date else None,
            "doctor": {
                "id": scan.doctor.id,
                "name": scan.doctor.name,
                "email": scan.doctor.email,
                "phone": scan.doctor.phone,
            } if scan.doctor else None,
        })
    return out


@router.delete("/patient-scans/{scan_id}")
def delete_patient_tumor_scan_request(
    scan_id: int,
    db: Session = Depends(get_db),
    current=Depends(role_required("patient")),
):
    """
    Remove a patient request (tumor or Alzheimer) for the logged-in patient:
    diagnoses, reports, PDFs, result images, and stored scan assets.
    The case disappears from patient and doctor dashboards.
    """
    scan = db.query(MRIScan).filter(MRIScan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    if scan.patient_id is None or int(scan.patient_id) != int(current.id):
        raise HTTPException(status_code=403, detail="You can only delete your own scans.")
    diagnoses = db.query(Diagnosis).filter(Diagnosis.scan_id == scan_id).all()
    for dx in diagnoses:
        for rep in db.query(Report).filter(Report.diagnosis_id == dx.id).all():
            _unlink_if_under_data(rep.pdf_path)
            _unlink_if_under_data(rep.file_path)
            db.delete(rep)
        _unlink_diagnosis_result_assets(dx)
        db.delete(dx)

    scan_dir = _scan_storage_dir(scan.id, scan.scan_kind)
    if os.path.isdir(scan_dir):
        shutil.rmtree(scan_dir, ignore_errors=True)
    kind_root = os.path.join(SCANS_UPLOAD_DIR, _scan_storage_kind(scan.scan_kind))
    staging = os.path.join(kind_root, f"{scan.id}.staging")
    if os.path.isdir(staging):
        shutil.rmtree(staging, ignore_errors=True)

    _unlink_if_under_data((scan.file_path or "").strip())

    db.delete(scan)
    db.commit()
    return {"ok": True, "id": scan_id}


@router.delete("/patient-scans/{scan_id}/delete")
def delete_patient_scan_request_alias(
    scan_id: int,
    db: Session = Depends(get_db),
    current=Depends(role_required("patient")),
):
    """
    Compatibility alias used by mobile patient dashboard delete action.
    Deletes tumor and Alzheimer requests from both patient and doctor queues.
    """
    return delete_patient_tumor_scan_request(scan_id=scan_id, db=db, current=current)


@router.get("/doctor-requests")
def get_doctor_requests(
    db: Session = Depends(get_db),
    current = Depends(role_required("doctor"))
):
    """Get all MRI scan requests for the clinic doctor workflow."""
    scans = (
        db.query(MRIScan)
        .options(joinedload(MRIScan.patient), joinedload(MRIScan.doctor))
        .filter(
            MRIScan.patient_id.isnot(None),
            MRIScan.doctor_id == current.id,
            MRIScan.status.in_(WORKFLOW_STATUSES),
        )
        .order_by(MRIScan.sent_date.desc(), MRIScan.upload_date.desc(), MRIScan.id.desc())
        .all()
    )
    
    # Enrich with diagnosis and report data
    result = []
    for scan in scans:
        fp = (scan.file_path or "").strip()
        filename = os.path.basename(fp) if fp else ""
        file_url = _public_uploads_url(fp) if fp else None
        sk = getattr(scan, "scan_kind", None) or "mri"
        scan_dict = {
            "id": scan.id,
            "scan_id": scan.id,
            "patient_id": scan.patient_id,
            "doctor_id": scan.doctor_id,
            "file_path": scan.file_path,
            "file_name": filename,
            "file_url": file_url,
            "scan_kind": sk,
            "status": scan.status.value if hasattr(scan.status, "value") else scan.status,
            "upload_date": scan.upload_date.isoformat() if scan.upload_date else None,
            "sent_date": scan.sent_date.isoformat() if scan.sent_date else None,
            "diagnosis": None,
            "patient": {
                "id": scan.patient.id,
                "name": scan.patient.name,
                "email": scan.patient.email,
                "age": scan.patient.age,
                "phone": scan.patient.phone,
            } if scan.patient else None,
        }
        
        # Latest diagnosis for this scan (multiple rows possible if re-run)
        diagnosis = (
            db.query(Diagnosis)
            .filter(Diagnosis.scan_id == scan.id)
            .order_by(Diagnosis.id.desc())
            .first()
        )
        if diagnosis:
            report = db.query(Report).filter(Report.diagnosis_id == diagnosis.id).first()
            probs = None
            if diagnosis.model_meta:
                try:
                    raw_meta = json.loads(diagnosis.model_meta)
                    if isinstance(raw_meta, dict) and raw_meta.get("report_type") == "segmentation_pdf":
                        probs = None
                    elif isinstance(raw_meta, dict) and "probs" in raw_meta:
                        probs = raw_meta.get("probs")
                    else:
                        probs = raw_meta
                except json.JSONDecodeError:
                    probs = None
            scan_dict["diagnosis"] = {
                "id": diagnosis.id,
                "prediction": diagnosis.prediction,
                "confidence": diagnosis.confidence,
                "model_version": diagnosis.model_version,
                "model_probs": probs,
                "report": {
                    "id": report.id if report else None,
                    "summary": report.summary if report else None,
                    "recommendation": report.recommendation if report else None,
                    "download_url": f"/reports/{report.id}" if report and (report.pdf_path or report.file_path) else None,
                } if report else None,
            }
        
        result.append(scan_dict)
    
    return result


@router.post("/scan/{scan_id}/replace-file")
def replace_patient_scan_file(
    scan_id: int,
    t1c: Optional[UploadFile] = File(default=None),
    t1n: Optional[UploadFile] = File(default=None),
    t2f: Optional[UploadFile] = File(default=None),
    t2w: Optional[UploadFile] = File(default=None),
    db: Session = Depends(get_db),
    current=Depends(role_required("doctor", "admin")),
):
    """Doctor replaces on-disk MRI for an assigned patient scan (after downloading / QC on PC)."""
    scan = db.query(MRIScan).filter(MRIScan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    if scan.patient_id is None:
        raise HTTPException(status_code=400, detail="Only patient-linked scans can be updated here")
    if scan.doctor_id != current.id and (current.role or "").lower() != "admin":
        raise HTTPException(status_code=403, detail="You are not assigned to this scan")
    if scan.status not in (ScanStatus.sent, ScanStatus.analyzed):
        raise HTTPException(
            status_code=400,
            detail="Replace is allowed for scans in 'sent' or 'analyzed' status only",
        )

    files_by_modality = {
        "t1c": t1c,
        "t1n": t1n,
        "t2f": t2f,
        "t2w": t2w,
    }
    _validate_all_modalities_present(files_by_modality)
    fpath = _save_modalities_to_scan_dir(scan.id, files_by_modality, scan.scan_kind)

    scan.file_path = fpath
    db.add(scan)
    db.commit()
    db.refresh(scan)

    return {
        "ok": True,
        "scan_id": scan.id,
        "file_name": os.path.basename(fpath),
        "file_url": _public_uploads_url(fpath),
    }
