FROM debian:testing-slim 

# Install deps
RUN apt-get update && \
    apt-get install -y build-essential cmake git libx11-dev libopencv-core-dev \
                       python3-opencv vim python3-dev swig x11-xserver-utils \
                       python3-serial v4l-utils

# Install libwallaby
COPY build_libwallaby.sh kipr_c.i /
RUN sh /build_libwallaby.sh

# Run c4 bot
CMD /c4/scripts/run.sh
