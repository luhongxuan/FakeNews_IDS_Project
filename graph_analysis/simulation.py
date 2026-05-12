import networkx as nx
import os
import sys
import matplotlib.pyplot as plt
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
from graph_analysis.builder import build_graph_from_pheme
from dotenv import load_dotenv

load_dotenv()

if os.path.exists("/home/luhongxuan/FakeNews_IDS_Project"):
    sys.path.append("/home/luhongxuan/FakeNews_IDS_Project")

def run_sir_simulation(
    G: nx.Graph,
    beta: float = 0.1,
    gamma: float = 0.05,
    infected_nodes: list = None,
    iterations: int = 50
) -> dict:
    """
    跑 SIR 模擬
    回傳每一輪的 S/I/R 數量變化
    """
    UG = G.to_undirected() if G.is_directed() else G

    model = ep.SIRModel(UG)

    cfg = mc.Configuration()
    cfg.add_model_parameter("beta", beta)
    cfg.add_model_parameter("gamma", gamma)

    if infected_nodes:
        cfg.add_model_parameter("fraction_infected", 0)
        cfg.add_model_initial_configuration("Infected", infected_nodes)
    else:
        cfg.add_model_parameter("fraction_infected", 0.05)

    model.set_initial_status(cfg)

    iterations_result = model.iteration_bunch(iterations)

    trends = model.build_trends(iterations_result)

    return {
        "iterations": iterations_result,
        "trends": {
            "susceptible": trends[0]["trends"]["node_count"][0],
            "infected":    trends[0]["trends"]["node_count"][1],
            "removed":     trends[0]["trends"]["node_count"][2],
        },
        "total_nodes": UG.number_of_nodes(),
        "peak_infected": max(trends[0]["trends"]["node_count"][1]),
    }


# load_dotenv()

# PHEME_PATH = os.getcwd() + os.getenv("PHEME_PATH")
# G = build_graph_from_pheme(PHEME_PATH, "charliehebdo")
# result = run_sir_simulation(G, beta=0.1, gamma=0.05, iterations=50)
# trends = result["trends"]

# x = range(len(trends["susceptible"]))

# plt.figure(figsize=(10, 5))
# plt.plot(x, trends["susceptible"], label="Susceptible", color="blue")
# plt.plot(x, trends["infected"],    label="Infected",    color="red")
# plt.plot(x, trends["removed"],     label="Removed",     color="green")

# plt.axvline(
#     x=trends["infected"].index(max(trends["infected"])),
#     color="red", linestyle="--", alpha=0.5,
#     label=f"Peak Infected: {result['peak_infected']} nodes"
# )

# plt.xlabel("Iteration")
# plt.ylabel("Node Count")
# plt.title("SIR Simulation — charliehebdo Event")
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.savefig("/home/luhongxuan/FakeNews_IDS_Project/data/sir_simulation.png", dpi=150, bbox_inches="tight")
# plt.close()