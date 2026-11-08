"""Example external controller for the ice-cream digital twin (as a ROS bridge node would be).

The simulation runs in the Rust server (`icecream web`, started by `just dev`); this script connects to
it like the browser does, asks for a state stream in ROS REP-103 coordinates (x forward, y left, z up,
metres; quaternions [x, y, z, w]) and scoops vanilla, lifts, "eats", then scoops mint. Open the page
alongside to watch it.

    uv run --with websockets tools/ws_example_client.py [ws://localhost:8765/ws]
"""

import asyncio
import json
import math
import sys

import websockets

BOWL_X = {"vanilla": -0.18, "strawberry": 0.0, "mint": 0.18}
# The SO-101 stands behind the bowls at ROS (0, 0.28, 0); its pan axis is ~3.9 cm in front of that.
PAN_AXIS = (0.0, 0.241)


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def spoon_quat(x, y, pitch):
    """Handle pointing at the arm (yaw about ROS z), cup tipped nose-down by `pitch` (about ROS -Y)."""
    yaw = math.atan2(PAN_AXIS[1] - y, PAN_AXIS[0] - x)
    yaw_q = [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]
    tilt_q = [0.0, -math.sin(pitch / 2), 0.0, math.cos(pitch / 2)]
    return quat_mul(yaw_q, tilt_q)


def scoop(bowl):
    x = BOWL_X[bowl]
    # unit vector from the bowl centre toward the arm: dig on the far side, drag toward the arm
    dx, dy = PAN_AXIS[0] - x, PAN_AXIS[1]
    n = math.hypot(dx, dy)
    ux, uy = dx / n, dy / n

    def at(s, z):
        return [x + ux * s, uy * s, z]

    steps = [(at(-0.02, 0.14), 0.5, 1.5), (at(-0.015, 0.086), 0.5, 2.5), (at(0.025, 0.088), 0.0, 2.5), (at(0.03, 0.16), 0.0, 2.0)]
    return [(p, spoon_quat(p[0], p[1], pitch), secs) for p, pitch, secs in steps]


async def drive(ws):
    latest = {}

    async def reader():
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "hello":
                print("twin says hello, parameter groups:", list(msg["params"]))
            elif msg.get("type") == "state":
                latest.update(msg)

    task = asyncio.create_task(reader())
    await ws.send(json.dumps({"type": "config", "stream": "state", "frame": "ros", "rateHz": 20}))

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
    # finish with a joint-space command (SO-101 URDF joint order, degrees)
    await ws.send(json.dumps({"type": "set_joints", "joints": [0, -90, 50, 70, 0], "unit": "deg"}))
    print("done; bowls:", {k: round(v, 1) for k, v in latest.get("bowlGrams", {}).items()})
    task.cancel()


async def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8765/ws"
    async with websockets.connect(url) as ws:
        print("connected to", url)
        await drive(ws)


if __name__ == "__main__":
    asyncio.run(main())
