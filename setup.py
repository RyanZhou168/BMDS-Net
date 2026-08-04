from setuptools import setup, find_packages

setup(
    name="bmds_net",
    version="1.0.0",
    description="BMDS-Net for deployment-aware multi-modal brain tumor segmentation",
    author="Yan Zhou",
    author_email="1727663150@qq.com",
    url="https://github.com/RyanZhou168/BMDS-Net",
    packages=find_packages(exclude=["tests", "tools", "configs"]),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "monai>=1.3.0",
        "nibabel",
        "torchio",
        "tqdm",
        "pyyaml"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
