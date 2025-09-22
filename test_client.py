import asyncio
import websockets

async def test_ws():
    uri = "ws://127.0.0.1:8001/ws/audio"
    async with websockets.connect(uri) as websocket:
        print("✅ Connected to server")
        await websocket.send("hello")
        msg = await websocket.recv()
        print("Received:", msg)

asyncio.run(test_ws())

