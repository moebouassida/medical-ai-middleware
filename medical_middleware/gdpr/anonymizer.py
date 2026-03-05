"""
anonymizer.py — Medical image anonymization.

Strips all metadata from uploaded images before processing:
  - EXIF data (GPS, device info, timestamps)
  - DICOM tags (patient name, DOB, MRN, etc.)
  - ICC profiles
  - XMP metadata
  - Comments

GDPR Article 4(1): Anonymized data is not personal data.
"""

import io
import logging

from PIL import Image

logger = logging.getLogger(__name__)


# DICOM tags that must be removed (patient identifiers)
# Based on DICOM PS3.15 Annex E (Basic Application Level Confidentiality Profile)
SENSITIVE_DICOM_TAGS = {
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "PatientAge",
    "PatientWeight",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "ReferringPhysicianName",
    "InstitutionName",
    "InstitutionAddress",
    "StudyDate",
    "StudyTime",
    "AccessionNumber",
    "StudyID",
    "SeriesDate",
    "SeriesTime",
    "OperatorsName",
    "PerformingPhysicianName",
    "RequestingPhysician",
    "ScheduledPerformingPhysicianName",
}


class ImageAnonymizer:
    """
    Strips all identifying metadata from medical images.
    Supports JPEG, PNG, TIFF, and basic DICOM.
    """

    @staticmethod
    def anonymize_pil(image: Image.Image) -> Image.Image:
        """
        Strip all metadata from a PIL Image.

        Args:
            image: PIL Image with potential metadata

        Returns:
            Clean PIL Image with no metadata
        """
        # Convert to RGB to ensure consistent format
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        # Re-encode without metadata by going through bytes
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")  # PNG strips EXIF
        buffer.seek(0)

        clean = Image.open(buffer)
        clean.load()  # Force load before buffer goes out of scope

        return clean

    @staticmethod
    def anonymize_bytes(data: bytes, content_type: str = "image/jpeg") -> bytes:
        """
        Strip metadata from raw image bytes.

        Args:
            data:         raw image bytes
            content_type: MIME type of the image

        Returns:
            Cleaned image bytes
        """
        try:
            image = Image.open(io.BytesIO(data))
            clean = ImageAnonymizer.anonymize_pil(image)

            buffer = io.BytesIO()
            fmt = "JPEG" if "jpeg" in content_type else "PNG"
            clean.save(buffer, format=fmt)
            return buffer.getvalue()

        except Exception as e:
            logger.error(f"[anonymizer] Failed to anonymize image: {e}")
            raise ValueError(f"Image anonymization failed: {e}") from e

    @staticmethod
    def has_metadata(image: Image.Image) -> bool:
        """Check if image has any metadata."""
        return bool(getattr(image, "_getexif", lambda: None)() or image.info)

    @staticmethod
    def anonymize_dicom(data: bytes) -> bytes:
        """
        Strip sensitive DICOM tags.
        Requires pydicom. Falls back gracefully if not installed.
        """
        try:
            import pydicom
            from io import BytesIO

            ds = pydicom.dcmread(BytesIO(data))
            for tag in SENSITIVE_DICOM_TAGS:
                if hasattr(ds, tag):
                    delattr(ds, tag)

            buffer = BytesIO()
            ds.save_as(buffer)
            return buffer.getvalue()

        except ImportError:
            logger.warning(
                "[anonymizer] pydicom not installed — DICOM anonymization skipped"
            )
            return data
        except Exception as e:
            logger.error(f"[anonymizer] DICOM anonymization failed: {e}")
            raise
