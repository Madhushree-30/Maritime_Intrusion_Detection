
import pandas as pd
import os


AGENT_FILE = (
    "/content/drive/MyDrive/ResearchProject/"
    "agentic_ai/outputs/agent_decisions_with_frame.csv"
)

OUTPUT_FILE = (
    "/content/drive/MyDrive/ResearchProject/"
    "agentic_ai/outputs/agent_decisions_clean.csv"
)


print("=" * 70)
print("CLEANING AGENT DECISION DUPLICATES")
print("=" * 70)


# ============================================================
# LOAD
# ============================================================

agent = pd.read_csv(AGENT_FILE)

print("\nOriginal Agent shape:")
print(agent.shape)


# ============================================================
# CHECK FRAME
# ============================================================

if "frame" not in agent.columns:
    raise ValueError(
        "Agent file does not contain frame column."
    )

if "track_id" not in agent.columns:
    raise ValueError(
        "Agent file does not contain track_id column."
    )


# ============================================================
# REMOVE ROWS WITHOUT FRAME
# ============================================================

before = len(agent)

agent = agent.dropna(
    subset=["frame", "track_id"]
).copy()

after = len(agent)

print("\nRows before removing missing frame:", before)
print("Rows after removing missing frame:", after)
print("Rows removed:", before - after)


# ============================================================
# NORMALIZE KEY TYPES
# ============================================================

agent["frame"] = pd.to_numeric(
    agent["frame"],
    errors="coerce"
)

agent["track_id"] = pd.to_numeric(
    agent["track_id"],
    errors="coerce"
)

agent = agent.dropna(
    subset=["frame", "track_id"]
).copy()

agent["frame"] = agent["frame"].astype(int)
agent["track_id"] = agent["track_id"].astype(int)


# ============================================================
# SORT
# ============================================================

agent = agent.sort_values(
    ["frame", "track_id"]
).reset_index(drop=True)


# ============================================================
# DUPLICATE ANALYSIS
# ============================================================

duplicate_rows = agent.duplicated(
    subset=["frame", "track_id"],
    keep=False
)

print("\nDuplicate rows:")
print(duplicate_rows.sum())

print(
    "\nDuplicate key count:"
)

print(
    agent.loc[
        duplicate_rows,
        ["frame", "track_id"]
    ]
    .drop_duplicates()
    .shape[0]
)


# ============================================================
# SHOW DUPLICATES
# ============================================================

print("\nSample duplicate keys:")

duplicates = (
    agent.loc[
        duplicate_rows,
        ["frame", "track_id"]
    ]
    .drop_duplicates()
    .head(20)
)

print(
    duplicates.to_string(index=False)
)


# ============================================================
# KEEP ONE AGENT DECISION PER
# (FRAME, TRACK_ID)
# ============================================================

agent_clean = (
    agent
    .drop_duplicates(
        subset=["frame", "track_id"],
        keep="first"
    )
    .reset_index(drop=True)
)


# ============================================================
# VERIFY
# ============================================================

remaining_duplicates = agent_clean.duplicated(
    subset=["frame", "track_id"]
).sum()

print("\n" + "=" * 70)
print("AFTER CLEANING")
print("=" * 70)

print("\nOriginal rows:", len(agent))
print("Clean rows:", len(agent_clean))
print("Rows removed:", len(agent) - len(agent_clean))

print(
    "Remaining duplicate keys:",
    remaining_duplicates
)

print(
    "Unique frame/track keys:",
    agent_clean[
        ["frame", "track_id"]
    ]
    .drop_duplicates()
    .shape[0]
)


if remaining_duplicates != 0:
    raise ValueError(
        "Duplicate frame/track keys still exist."
    )


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

agent_clean.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nSaved cleaned Agent file:")
print(OUTPUT_FILE)

print("\nFinal columns:")
print(agent_clean.columns.tolist())

print("\nSample:")
print(
    agent_clean[
        [
            "frame",
            "track_id",
            "MMSI",
            "intrusion_type",
            "severity",
            "decision",
            "action"
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print("\n" + "=" * 70)
print("AGENT CLEANING COMPLETE")
print("=" * 70)
