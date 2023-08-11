---
title: "A Simple Python Memory Usage Profiler"
description: "python profiler"
pubDate: "June 10 2022"
heroImage: "/Earth.jpg"
---

## Install Linux Perf

```bash
sudo apt-get install linux-tools

# run perf without root permission
sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'
```


## Create Context Manager


```python

from contextlib import contextmanager  
from os import getpid  
from resource import RUSAGE_SELF, getrusage  
from signal import SIGINT  
from subprocess import Popen  
from time import sleep, time  
  
events = [  
    "instructions",  
    "cache-references",  
    "cache-misses",  
]  
  
  
@contextmanager  
def track_peak_memory():  
    process = Popen(["perf", "stat", "-p", str(getpid()), "-e", ",".join(events)])  
    sleep(0.1)  
    start_time = time()  
    try:  
        yield  
    finally:  
        print(f"Run time (s): {time() - start_time}")  
        print("Peak memory (MiB):", int(getrusage(RUSAGE_SELF).ru_maxrss / 1024))  
        process.send_signal(SIGINT)  
  
  
if __name__ == "__main__":  
    with track_peak_memory():  
  
        class Test:  
            a: int = 1  
  
        num = 1_000_000_000  
        result = [Test()] * num  
  
    # Run time (s): 4.494251489639282  
    # Peak memory (MiB): 7503

```