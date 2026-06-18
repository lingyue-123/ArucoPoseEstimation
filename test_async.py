import asyncio
import logging
import time


logger = logging.getLogger("TestAsync")


async def async_test():
    print("Async test started.")
    logger.info("Starting async test...")
    times = 0
    while times < 3:
        time.sleep(1)
        times += 1
        logger.info(f"gripper is opening... {times} seconds")
        print(f"gripper is opening... {times} seconds")
    logger.info("Async test completed.")
    print("Async test completed.")

async def main():
    
    task = asyncio.create_task(async_test())
    logger.info("Main function continues while async test is running...")
    print("Main function continues while async test is running...")
    await task
    

if __name__ == "__main__":
    asyncio.run(main())
