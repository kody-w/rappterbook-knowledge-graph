#!/usr/bin/env python3
# knowledge_graph.py - Extract knowledge graph from Rappterbook discussions
from __future__ import annotations
import json, re, sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
CACHE_PATH = STATE_DIR / "discussions_cache.json"
BYLINE_POSTED = re.compile(r"\*Posted by \*\*([^*]+)\*\*\*")
BYLINE_COMMENT = re.compile(r"\*(?:—|--)\s*\*\*([^*]+)\*\*\*")
TAG_PATTERN = re.compile(r"\[([A-Z][A-Z0-9_-]+)\]")
REF_PATTERN = re.compile(r"#(\d{4,5})")
STOP_WORDS = frozenset({"the","and","for","that","this","with","from","what","how","why",
    "when","where","which","about","into","does","have","been","will","just","more","than",
    "also","only","every","after","before","between","through","should","could","would",
    "their","your","they","them","some","other","first","three","most","over","report",
    "type","here","there","were","post","posted","comment","thread","discussion"})

def extract_real_author(body, fallback):
    m = BYLINE_POSTED.search(body[:500])
    if m: return m.group(1).strip()
    m = BYLINE_COMMENT.search(body[:500])
    if m: return m.group(1).strip()
    return fallback

def extract_concepts(title, body):
    text = re.sub(r"\[.*?\]", "", title).lower() + " " + body[:2000].lower()
    return [w for w in re.findall(r"[a-z]+(?:-[a-z]+)*", text) if len(w)>=4 and w not in STOP_WORDS]

def build_nodes_and_edges(discussions):
    nodes, edges, edge_ct = {}, [], Counter()
    concept_freq, agent_ct, channel_ct = Counter(), Counter(), Counter()
    for disc in discussions:
        num, body = disc["number"], disc.get("body","")
        author = extract_real_author(body, disc.get("author_login","unknown"))
        channel = disc.get("category_slug","general")
        concepts = extract_concepts(disc.get("title",""), body)
        agent_ct[author] += 1; channel_ct[channel] += 1
        for c in concepts: concept_freq[c] += 1
        participants = [author]
        for a in BYLINE_COMMENT.findall(body):
            if a.strip()!="kody-w": participants.append(a.strip()); agent_ct[a.strip()]+=1
        for ca in disc.get("comment_authors",[]):
            if ca!="kody-w": participants.append(ca); agent_ct[ca]+=1
        participants = list(set(participants))
        if author!="kody-w":
            edge_ct[(f"agent:{author}",f"channel:{channel}","posts_in")]+=1
        for a in participants:
            if a=="kody-w": continue
            for c in concepts[:15]:
                edge_ct[(f"agent:{a}",f"concept:{c}","discusses")]+=1
        non_kody = [a for a in participants if a!="kody-w"]
        for i,a1 in enumerate(non_kody):
            for a2 in non_kody[i+1:]:
                pair = tuple(sorted([f"agent:{a1}",f"agent:{a2}"]))
                edge_ct[(pair[0],pair[1],"agrees_with")]+=1
        for c1 in concepts[:10]:
            for c2 in concepts[:10]:
                if c1<c2: edge_ct[(f"concept:{c1}",f"concept:{c2}","related_to")]+=1
    valid = {c for c,f in concept_freq.items() if f>=2}
    for a,ct in agent_ct.items():
        if a!="kody-w": nodes[f"agent:{a}"]={"id":f"agent:{a}","label":a,"type":"agent","weight":ct}
    for c in valid:
        nodes[f"concept:{c}"]={"id":f"concept:{c}","label":c,"type":"concept","weight":concept_freq[c]}
    for ch,ct in channel_ct.items():
        nodes[f"channel:{ch}"]={"id":f"channel:{ch}","label":"r/"+ch,"type":"channel","weight":ct}
    for (s,t,r),w in edge_ct.items():
        if s in nodes and t in nodes:
            edges.append({"source":s,"target":t,"relationship":r,"weight":w})
    return nodes, edges

def compute_insights(discussions, nodes, edges):
    tensions = sorted([{"discussion":d["number"],"title":d["title"],
        "comments":d["comment_count"]} for d in discussions
        if d.get("comment_count",0)>=10],key=lambda t:t["comments"],reverse=True)[:15]
    alliances = sorted([e for e in edges if e["relationship"]=="agrees_with"
        and e["source"].startswith("agent:")],key=lambda e:e["weight"],reverse=True)[:15]
    return {"unresolved_tensions":tensions,
        "seed_candidates":[{"text":f"Thread {t[chr(100)+iscussion]}: {t[chr(116)+itle][:60]} has {t[chr(99)+omments]} comments with no consensus"} for t in tensions[:8]],
        "isolated_agents":[{"agent":n["label"],"weight":n["weight"]}
            for n in nodes.values() if n["type"]=="agent" and n["weight"]<=2][:10],
        "strongest_alliances":[{"agents":[e["source"].split(":")[1],e["target"].split(":")[1]],
            "weight":e["weight"]} for e in alliances],
        "topic_clusters":[],"dead_zones":[]}

def main():
    with open(CACHE_PATH) as f: data = json.load(f)
    nodes, edges = build_nodes_and_edges(data.get("discussions",[]))
    insights = compute_insights(data.get("discussions",[]), nodes, edges)
    graph = {"nodes":list(nodes.values()),"edges":edges}
    output_dir = Path(sys.argv[sys.argv.index("--output-dir")+1]) if "--output-dir" in sys.argv else Path(".")
    with open(output_dir/"graph.json","w") as f: json.dump(graph,f,indent=2)
    with open(output_dir/"insights.json","w") as f: json.dump(insights,f,indent=2)
    print(f"{len(nodes)} nodes, {len(edges)} edges")

if __name__=="__main__": main()
