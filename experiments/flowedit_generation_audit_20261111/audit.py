"""CPU audit of saved FlowEdit states; never load or execute an image model.

Reconstructed queries use CPU BF16 operations. They are not saved GPU queries,
and no cross-device bitwise equivalence is asserted.
"""

import hashlib
import json
import math
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OLD = ROOT / "experiments/sota_inversion_20261105/outputs/comparison_v1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rms(tensor):
    return float(tensor.float().square().mean().sqrt())


def main():
    torch.set_num_threads(4)
    protocol = json.loads((OLD / "protocol.json").read_text())
    checkpoint = torch.load(OLD / "checkpoint.pt", map_location="cpu", weights_only=True)
    x = checkpoint["states"]["source"]
    assert x.dtype == torch.bfloat16 and tuple(x.shape) == (1, 4096, 64)
    code_checks = {}
    for key, relative in [
        ("script_sha256", "experiments/sota_inversion_20261105/run_comparison.py"),
        ("common_sha256", "experiments/sota_inversion_20261105/common.py"),
        ("official_solver_sha256", "vendor/selfix/src/flux/sampling.py"),
        ("official_layers_sha256", "vendor/selfix/src/flux/modules/layers.py")]:
        code_checks[relative] = digest(ROOT / relative) == protocol[key]
    assert all(code_checks.values())

    config_path = ROOT / "pretrained_models/flux1-dev/scheduler/scheduler_config.json"
    config = json.loads(config_path.read_text())
    mu = config["base_shift"] + (
        (x.shape[1] - config["base_image_seq_len"]) *
        (config["max_shift"] - config["base_shift"]) /
        (config["max_image_seq_len"] - config["base_image_seq_len"]))
    recipes = {}
    for recipe in protocol["flowedit_recipes"]:
        assert recipe["n_avg"] == 1 and recipe["n_min"] == 0
        trace = checkpoint["traces"][recipe["name"]]
        states, times = trace["states"], trace["timesteps"]
        unshifted = torch.linspace(1, 0, recipe["T"] + 1)
        rebuilt = math.exp(mu) / (math.exp(mu) + (1 / unshifted - 1))
        expected = rebuilt[recipe["T"] - recipe["n_max"]:].tolist()
        assert times == expected
        assert len(states) == len(times) == recipe["n_max"] + 1
        assert torch.equal(states[0], x)
        assert checkpoint["counts"][recipe["name"]] == 2 * recipe["n_max"]
        rng = torch.Generator("cpu").manual_seed(protocol["seed"])
        rows = []
        for i, (t, tn) in enumerate(zip(times[:-1], times[1:])):
            y = states[i]
            noise = torch.randn(x.shape, generator=rng, dtype=torch.bfloat16)
            qs = (1 - t) * x + t * noise
            qt = y + qs - x
            d = y.float() - x.float()
            source_float = (1 - t) * x.float() + t * noise.float()
            target_float = source_float + d
            geometric_defect = qt.float() - qs.float() - d
            delta = states[i+1].float() - y.float()
            rows.append(dict(
                step=i, t=t, next_t=tn, positive_step=t-tn,
                model_bf16_t=float(torch.tensor(t, dtype=torch.bfloat16)),
                model_time_abs_rounding=abs(float(torch.tensor(t, dtype=torch.bfloat16))-t),
                source_query_rounding_rms=rms(qs.float()-source_float),
                target_query_rounding_rms=rms(qt.float()-target_float),
                parallelogram_rounding_rms=rms(geometric_defect),
                parallelogram_nonzero_fraction=float((geometric_defect != 0).float().mean()),
                state_step_rms=rms(delta),
                unchanged_state_coordinate_fraction=float((delta == 0).float().mean()),
                edit_displacement_rms=rms(d)))
        recipes[recipe["name"]] = dict(
            steps=len(rows), branch_calls=checkpoint["counts"][recipe["name"]],
            schedule_start=times[0], schedule_end=times[-1],
            step_min=min(r["positive_step"] for r in rows),
            step_max=max(r["positive_step"] for r in rows),
            max_model_time_rounding=max(r["model_time_abs_rounding"] for r in rows),
            first_step=rows[0], per_step=rows)

    # BF16 violation of a real-arithmetic identity, using exactly representable inputs.
    unit = torch.tensor([1.], dtype=torch.bfloat16)
    epsilon = torch.tensor([-.9921875], dtype=torch.bfloat16)
    qs = .5 * unit + .5 * epsilon
    qt = unit + qs - unit
    assert float(qs) == 1/256 and float(qt) == 0
    # FP32 accumulator cannot retain an increment after writing it back to BF16.
    velocity = torch.tensor([.125], dtype=torch.bfloat16)
    intermediate = unit.float() + .01 * velocity
    stored = intermediate.to(torch.bfloat16)
    assert float(intermediate) > 1 and float(stored) == 1

    # Target-query displacement is d, not the normal noising displacement (1-t)d.
    xx, yy, ee, tt = 2., 3., -.4, .6
    source = (1-tt)*xx + tt*ee
    target = yy + source - xx
    noised_y = (1-tt)*yy + tt*ee
    assert abs(target-noised_y-tt*(yy-xx)) < 1e-12

    # A time-dependent random field with zero one-step mean can yield nonzero
    # expected terminal bias under a nonlinear subsequent update.
    signs = torch.tensor([-1., 1.], dtype=torch.float64)
    terminal = signs + signs.square()
    assert float(terminal.mean()) == 1

    output = dict(
        date="2026-11-11", new_image_model_forwards=0,
        scope="saved-state CPU query reconstruction; not GPU replay or quality attribution",
        input_files={str(p.relative_to(ROOT)): digest(p) for p in
                     [OLD/"checkpoint.pt", OLD/"protocol.json", config_path]},
        current_code_matches_recorded_protocol=code_checks,
        source_shape=list(x.shape), source_dtype=str(x.dtype), schedule_shift_mu=mu,
        recipes=recipes,
        analytic_checks=dict(
            bf16_query_identity=dict(source=float(qs), target=float(qt), clean_states_equal=True),
            fp32_then_bf16_lost_update=dict(intermediate=float(intermediate), stored=float(stored)),
            target_query_not_normal_noising=dict(actual=target, noised_current_y=noised_y,
                                                difference=target-noised_y),
            zero_mean_one_step_not_zero_terminal_bias=float(terminal.mean())),
        unmeasured=["raw model velocities", "actual GPU query tensors", "model approximation error",
                    "semantic contribution of each error", "cross-wrapper forward equivalence"])
    out = HERE / "audit.json"
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(out),
                         recipes={k:{kk:vv for kk,vv in v.items() if kk != "per_step"}
                                  for k,v in recipes.items()},
                         analytic_checks=output["analytic_checks"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
