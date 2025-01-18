import asyncio

async def compute(x,y,callback):
    print("开始计算") 
    await asyncio.sleep(0.5) # 模拟计算
    result = x * y
    callback(result)
    print("计算完成")


def print_result(result):
    print(f"The result is: {result}")

def square_result(result):
    print(f"The square result is: {result * result}")


async def another_task():
    print("开始另一个任务")
    await asyncio.sleep(1) # 模拟计算
    print("另一个任务完成") 


async def main():
    # await asyncio.gather(compute(1, 2, print_result), another_task())
    print("main 开始计算")
    task = asyncio.create_task(compute(1, 2, print_result))
    task2 = asyncio.create_task(another_task())

    await task
    await task2
    print("main 计算完成")

asyncio.run(main())
