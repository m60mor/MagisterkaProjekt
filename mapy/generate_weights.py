# generate_weights.py
import sys, sumolib

def weight(edge):
    prio = edge.getPriority()
    return 8.0 if prio >= 10 else 2.5   # tertiary vs. residential; service edges are filtered out below anyway

net = sumolib.net.readNet(sys.argv[1])
prefix = sys.argv[2]
for kind in ("src", "dst", "via"):
    with open(f"{prefix}.{kind}.xml", "w") as f:
        f.write('<edgedata>\n  <interval begin="0" end="10000">\n')
        for e in net.getEdges():
            if e.allows("passenger"):
                f.write(f'    <edge id="{e.getID()}" value="{weight(e):.2f}"/>\n')
        f.write("  </interval>\n</edgedata>\n")