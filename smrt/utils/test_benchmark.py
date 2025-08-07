import time
def test_time(benchmark):
    duration = 0.1
    benchmark(time.sleep,duration)