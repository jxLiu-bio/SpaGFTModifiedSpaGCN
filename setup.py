import setuptools

setuptools.setup(
    name="SpaGCN_SpaGFT", 
    version="1.0.0",
    author="Jian Hu, Jixin Liu",
    author_email="jxliu@mail.sdu.edu.cn",
    packages=setuptools.find_packages(),
    install_requires=["igraph","torch","pandas","numpy","scipy","scanpy","anndata","louvain","scikit-learn", "numba", "opencv-python"],
    #install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
