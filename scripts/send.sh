#!/usr/bin/env bash
DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
rsync -avz --delete "$DIR"/../ robot.rpivpn:c4/ 
