import time
def test_time(benchmark):
    duration = 0.0001
    benchmark(time.sleep,duration)