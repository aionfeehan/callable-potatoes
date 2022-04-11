import aiohttp
import asyncio
import concurrent.futures
import json
import time


async def test_service(session, url):
    option = {
        'expiry': 1,
        'maturity': 1.25,
        'strike': 0.01
    }
    async with session.post(url, json=option) as response:
        data = await response.text()
    return data


async def test_multiple_requests(url, n_total_requests=100):
    async with aiohttp.ClientSession() as session:
        post_tasks = [test_service(session, url) for k in range(n_total_requests)]
        res = await asyncio.gather(*post_tasks, return_exceptions=True)
    return res


async def multithread_main(threads=8):
    start = time.time()
    loop = asyncio.get_running_loop()
    n_total_requests = 100
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        result = await loop.run_in_executor(pool, test_multiple_requests, n_total_requests)
    end = time.time()
    total_time = round(end - start, 2)
    print(f'Handled {n_total_requests} requests from {threads} jobs in {total_time} seconds')
    return


async def main():
    start = time.time()
    n_total_requests = 20
    res = await test_multiple_requests("http://raspberrypi.local:8081/price", n_total_requests)
    end = time.time()
    total_time = round(end - start, 2)
    print(f'Handled {n_total_requests} requests {total_time} seconds')
    print(json.loads(res[0]))

if __name__ == '__main__':
    asyncio.run(main())
