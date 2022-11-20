#!/usr/bin/env bash
export DISPLAY=:0
if [ ! -f /.dockerenv ]; then
  DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
  cd "$DIR"
  docker-compose build
  xhost si:localuser:root >/dev/null
  docker-compose run c4 "$@"
  docker-compose down --remove-orphans
else
  ARCH=$(uname -m)
  export LD_PRELOAD="/usr/lib/$ARCH-linux-gnu/libX11.so /lib/$ARCH-linux-gnu/libpthread.so.0 /libwallaby/lib/libkipr.so /libwallaby/lib/_kipr.so"
  export PYTHONPATH="/libwallaby/lib"
  if [ -n "$*" ]; then
    exec "$@"
  else
    exec /c4/c4.py
  fi
fi
