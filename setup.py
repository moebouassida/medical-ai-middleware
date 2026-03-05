from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="medical-ai-middleware",
    version="1.1.0",
    author="Moez Bouassida",
    description="Production-grade GDPR, monitoring, and rate limiting middleware for medical AI APIs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/moebouassida/medical-ai-middleware",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "fastapi>=0.104.0",
        "starlette>=0.27.0",
        "Pillow>=10.0.0",
    ],
    extras_require={
        "monitoring": ["prometheus-client>=0.19.0"],
        "ratelimit":  ["slowapi>=0.1.9"],
        "dicom":      ["pydicom>=2.4.0"],
        "s3":         ["boto3>=1.34.0"],
        "all": [
            "prometheus-client>=0.19.0",
            "slowapi>=0.1.9",
            "pydicom>=2.4.0",
            "boto3>=1.34.0",
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "numpy>=1.24.0",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Security",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords=[
        "medical-ai", "GDPR", "compliance", "fastapi",
        "prometheus", "grafana", "rate-limiting", "security", "s3",
    ],
)
