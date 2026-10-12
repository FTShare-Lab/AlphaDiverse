import os


def model_threads():
    return max(1, int(os.getenv("ALPHADIVERSE_THREADS", "4")))


def model_eval_workers():
    return 1
