# Runtime environment for the grid-cells code (Banino et al. 2018).
# The code targets TensorFlow 1.x / Sonnet v1, which need Python <= 3.7;
# this image (TF 1.15.5, Python 3.6) is the last TF1 release.
FROM tensorflow/tensorflow:1.15.5-py3

RUN pip install --no-cache-dir \
    dm-sonnet==1.36 \
    tensorflow-probability==0.8.0 \
    scipy==1.5.4 \
    matplotlib==3.3.4

WORKDIR /workspace
