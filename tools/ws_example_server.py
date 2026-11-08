"""Example external controller for the ice-cream digital twin.

The browser page is the WebSocket *client*; this script is the server (as a ROS bridge node
would be). It scoops vanilla, lifts, "eats", then scoops mint, using ROS REP-103 coordinates
(x forward, y left, z up, metres; quaternions [x, y, z, w]).

    uv run --with websockets tools/ws_example_server.py
    # then in the page: Parameters > Digital-twin bridge > Connect   (or sim.connect())
"""

import asyncio
import json
import math

import websockets

BOWL_X = {"vanilla": -0.18, "strawberry": 0.0, "mint": 0.18}


def tilt_quat(rad):
    """Cup tipped toward +x (three.js rotation about +Z == ROS rotation about -Y)."""
    return [0.0, -math.sin(rad / 2), 0.0, math.cos(rad / 2)]


def scoop(bowl):
    x = BOWL_X[bowl]
    dig, level = tilt_quat(0.5), tilt_quat(0.0)
    return [
        ([x - 0.03, 0.0, 0.14], dig, 1.5),
        ([x - 0.02, 0.0, 0.085], dig, 2.5),  # push in
        ([x + 0.025, 0.0, 0.088], level, 2.5),  # drag + level the cup
        ([x + 0.025, 0.0, 0.16], level, 2.0),  # lift
    ]


async def handler(ws):
    latest = {}

    async def reader():
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "hello":
                print("twin connected, params:", list(msg["params"]))
            elif msg.get("type") == "state":
                latest.update(msg)

    task = asyncio.create_task(reader())
    await ws.send(json.dumps({"type": "config", "frame": "ros", "rateHz": 20}))

    async def run(bowl):
        for pos, quat, secs in scoop(bowl):
            await ws.send(json.dumps({"type": "set_target", "position": pos, "quaternion": quat}))
            await asyncio.sleep(secs)
            s = latest.get("spoon", {})
            print(
                f"{bowl:10s} pos={[round(v, 3) for v in s.get('position', [])]} "
                f"contact={latest.get('contact')} blocked={latest.get('blocked')} "
                f"cut={latest.get('cutForceN', 0):.1f}N on_spoon={latest.get('scoopedGrams', 0):.1f}g"
            )
        await ws.send(json.dumps({"type": "eat"}))
        await asyncio.sleep(0.3)
        print(f"ate it -> eaten total {latest.get('eatenGrams', 0):.1f} g")

    await run("vanilla")
    await run("mint")
    await ws.send(json.dumps({"type": "set_target", "position": [0.0, -0.02, 0.17], "quaternion": [0, 0, 0, 1]}))
    print("done; bowls:", {k: round(v, 1) for k, v in latest.get("bowlGrams", {}).items()})
    task.cancel()


async def main():
    async with websockets.serve(handler, "localhost", 8765):
        print("waiting for the twin on ws://localhost:8765 ...")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
