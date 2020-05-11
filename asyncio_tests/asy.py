import time
import requests
import asyncio

BASE_URL = "https://jsonplaceholder.typicode.com/"


# コルーチン関数
async def get_async(path: str) -> dict:
    print(f"/{path} async request")
    url = BASE_URL + path
    loop = asyncio.get_event_loop()
    # イベントループで実行
    res = await loop.run_in_executor(None, requests.get, url)
    print(f"/{path} async request done")
    return res.json()


def calc_time(fn):
    """関数の実行時間を計測するデコレータ"""
    def wrapper(*args, **kwargs):
        start = time.time()
        fn(*args, **kwargs)
        end = time.time()
        print(f"[{fn.__name__}] elapsed time: {end - start}")
        return
    return wrapper


@calc_time
def main_async():
    # イベントループを取得
    loop = asyncio.get_event_loop()
    # 非同期実行タスクを一つのFutureオブジェクトに
    tasks = asyncio.gather(
        get_async("posts"),
        get_async("comments"),
        get_async("albums"),
        get_async("photos"),
        get_async("todos"),
        get_async("users"),
    )
    # 非同期実行、それぞれが終わるまで
    results = loop.run_until_complete(tasks)  # resultsにFutureオブジェクトの戻り値
    print(results)
    return results


if __name__ == "__main__":
    print(main_async())  # None -> なんで
