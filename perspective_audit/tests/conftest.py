"""Settings shared by all the tests."""
try:
    import matplotlib

    matplotlib.use("Agg")           # the tests only save figures: never open a window (a Tk window fails on some runs, depending on the order of the tests)
except ImportError:                 # matplotlib is optional (the `plots` extra); the tests that need it skip themselves
    pass
