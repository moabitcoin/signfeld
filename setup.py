from setuptools import find_packages, setup

setup(
    name="synthetic-signs",
    version="0.0.0",
    author="DAS",
    url="https://gitlab.mobilityservices.io/am/roam/perception/experiments/"
    "synthetic-signs",
    description="Synthetic sign dataset generation and training tools",
    packages=find_packages(exclude=("notebooks", "resources", "scripts",
                                    "tests")),
    python_requires=">=3.6",
    scripts=[
        "bin/convert-gtsdb",
        "bin/detect-synthetic-signs",
        "bin/generate-label-map",
        "bin/generate-synthetic-dataset",
        "bin/generate-train-val-splits",
        "bin/train-synthetic-sign-detector",
        "bin/visualize-synthetic-sign-dataset",
        "bin/visualize-synthetic-sign-detections",
    ],
    setup_requires=[
        "Cython",
        "numpy",
    ],
    install_requires=[
        "numpy",
        "opencv-python",
        "Pillow",
        "pyyaml",
        "scikit-image",
        "scipy",
        "tqdm",
    ],
    extras_require=dict(trainer=[
        "torch>=1.3.0",
        "torchvision>=0.4.1",
        "pycocotools @ git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI",
        "detectron2 @ git+https://github.com/facebookresearch/detectron2.git",
    ], ))
