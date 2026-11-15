#!/usr/bin/env bash
# The only way the driver is ever started (launch.py and the watchdog both call this).
# driver.py takes an exclusive flock on driver.lock for its whole life, so a second launch
# exits at once instead of starting a second GPU job. Every fd is redirected so the ssh
# channel closes as soon as this script returns.
AUTO=/data/p3-moonlight-quality/autorun
mkdir -p "$AUTO/logs"
cd "$AUTO" || exit 1
setsid nohup python3 "$AUTO/driver.py" >>"$AUTO/driver.log" 2>&1 </dev/null &
exit 0
