"""Archive count gate, imported only during the consistency stage."""


def check(threads, replies, source_counts, extra, root):
    observed = (source_counts["threads"], source_counts["rumour_threads"],
                source_counts["events"], source_counts["self_replies"])
    if observed != (6425, 2402, 9, 2936):
        raise ValueError("source count gate failed")
    return {"passed": True, "all_threads": observed[0], "rumour_threads": observed[1],
            "events": observed[2], "self_replies": observed[3]}
