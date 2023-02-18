import numpy as np
import matplotlib.pyplot as plt
import json

file = f"1670750970.4062474/metrics_breakout.npy"

metrics = np.load(file, allow_pickle=True).item()
hypers = json.load(open(file.split("/")[0] + "/args.json", "r"))

episodes = metrics["episodes"]

plt.xlabel("Episode")
plt.ylabel("Learning Curve")

# Show every tenth episode as an xtick
plt.xticks(range(0, len(episodes), 100))

plt.title(
    f"Atari 2600 '" + hypers["environment"] + f"' trained over {len(episodes)} episodes"
)

plt.plot(
    range(len(episodes)),
    [enum / ep["cumulative_reward"] for enum, ep in enumerate(episodes)],
    color="red",
)

plt.show()
