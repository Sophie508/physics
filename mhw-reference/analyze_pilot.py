import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("run_directory")
args = parser.parse_args()
run_directory = Path(args.run_directory)
records = [json.loads(line) for line in (run_directory / "diagnostics.jsonl").read_text().splitlines()]


def summarize(block):
    result = {
        "start_time": block[0]["time"],
        "end_time": block[-1]["time"],
        "count": len(block),
    }
    for key in ["kinetic_energy", "zonal_energy_fraction", "particle_flux"]:
        values = np.asarray([row[key] for row in block], dtype=float)
        result[f"{key}_median"] = float(np.median(values))
        result[f"{key}_min"] = float(np.min(values))
        result[f"{key}_max"] = float(np.max(values))
        result[f"{key}_first"] = float(values[0])
        result[f"{key}_last"] = float(values[-1])
        result[f"{key}_linear_slope"] = float(
            np.polyfit([row["time"] for row in block], values, 1)[0]
        )
    positive_energy = np.asarray([row["kinetic_energy"] for row in block])
    result["kinetic_energy_log_slope"] = float(
        np.polyfit(
            [row["time"] for row in block], np.log(np.maximum(positive_energy, 1e-300)), 1
        )[0]
    )
    return result


count = len(records)
eighth = count // 8
previous = records[-2 * eighth : -eighth]
final = records[-eighth:]
previous_summary = summarize(previous)
final_summary = summarize(final)
relative_median_changes = {}
for key in ["kinetic_energy", "zonal_energy_fraction"]:
    left = previous_summary[f"{key}_median"]
    right = final_summary[f"{key}_median"]
    relative_median_changes[key] = abs(right - left) / max(abs(left), abs(right), 1e-12)

report = {
    "record_count": count,
    "eighth_record_count": eighth,
    "previous_eighth": previous_summary,
    "final_eighth": final_summary,
    "adjacent_eighth_relative_median_changes": relative_median_changes,
    "adjacent_eighth_20_percent_gate": all(
        value <= 0.20 for value in relative_median_changes.values()
    ),
    "note": "Trend slopes are descriptive because the preregistration did not define a numeric non-growth threshold.",
}
(run_directory / "analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
