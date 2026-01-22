FROM python:3.11-slim-bookworm

ARG DR2STAR_VERSION=unknown
ENV DR2STAR_VERSION=$DR2STAR_VERSION

# ~10Mb of AFNI tools.  latest pulled 20260102, created 2025-12-18
COPY --from=docker.io/afni/afni_make_build@sha256:5e0d8733ed277ea58b4a527e88bc10f62572ee63308d97a5e5e340d4423b3804 \
  /opt/afni/install/libmri.so \
  /opt/afni/install/libf2c.so \
  /opt/afni/install/3dBrickStat \
  /opt/afni/install/3dcalc \
  /opt/afni/install/3dinfo \
  /opt/afni/install/3dNotes \
  /opt/afni/install/3dROIstats \
  /opt/afni/install/3dTcat \
  /opt/afni/install/3dTstat \
  /usr/bin/

# depends read from 'ldd': libz libexpat
RUN apt-get update -qq && \
  apt-get install -qy --no-install-recommends \
    parallel \
    libexpat1 \
    zlib1g && \
  rm -rf /var/lib/apt/lists/* 

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN python3 -m pip install --no-cache-dir \
  numpy==1.26.4 \
  scipy==1.11.4 \
  pandas==2.2.3 \
  nibabel==5.3.2


COPY tat2 /usr/bin/
COPY dR2star /opt/dR2star/dR2star
ENV PYTHONPATH=/opt/dR2star
RUN chmod +x /usr/bin/tat2
ENTRYPOINT ["python3", "-m", "dR2star.run"]
