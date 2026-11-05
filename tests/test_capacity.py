from gymtosha.benchmarks.capacity import search_capacity


def test_capacity_doubles_until_first_failure():
    seen = []

    def probe(n):
        seen.append(n)
        return {"num_envs": n, "status": "passed" if n <= 16384 else "failed"}

    result = search_capacity(probe)
    assert seen == [4096, 8192, 16384, 32768]
    assert result["maximum_verified"] == 16384
    assert result["first_failed"] == 32768


def test_capacity_halves_from_failed_start():
    seen = []

    def probe(n):
        seen.append(n)
        return {"num_envs": n, "status": "passed" if n <= 1024 else "failed"}

    result = search_capacity(probe)
    assert seen == [4096, 2048, 1024]
    assert result["maximum_verified"] == 1024
    assert result["first_failed"] == 2048


def test_async_capacity_never_exceeds_32():
    seen = []
    result = search_capacity(
        lambda n: seen.append(n) or {"num_envs": n, "status": "passed"}, asynchronous=True
    )
    assert seen == [32]
    assert result["maximum_verified"] == 32
    assert result["capped"]
