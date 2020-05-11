import time
import requests

BASE_URL = "https://jsonplaceholder.typicode.com/"


def calc_time(fn):
    """関数の実行時間を計測するデコレータ"""
    def wrapper(*args, **kwargs):
        start = time.time()
        fn(*args, **kwargs)
        end = time.time()
        print(f"[{fn.__name__}] elapsed time: {end - start}")
        return
    return wrapper


def get_sync(path: str) -> dict:
    print(f"/{path} request")
    res = requests.get(BASE_URL + path)
    print(f"/{path} request done")
    return res.json()


@calc_time
def main_sync():
    data_ls = []
    paths = [
        "posts",
        "comments",
        "albums",
        "photos",
        "todos",
        "users",
    ]
    for path in paths:
        data_ls.append(get_sync(path))
    return data_ls


if __name__ == "__main__":
    main_sync()
