"""Exercises the same build/save function run/main.py calls, so the derived_latents build/save
pipeline can be run directly without going through main.py -- for manual runs, not CI (no CI is
set up for this project): either `uv run pytest run/tests/test_derived_latents.py -v`, or
`uv run python run/tests/test_derived_latents.py` to run the checks in order as a plain script.

Requires general/home_price/market's fit.json to already exist (run the three run/manual/build_*
scripts first if not) -- this makes no external API calls itself, only combines already-saved fits.
"""

from condobuyuq2026.input_deck import get_derived_latents, load_deck
from condobuyuq2026.reporting import print_derived_latent_build_report
from condobuyuq2026.utils.manual_utils import build_all_derived_latent_models, build_derived_latent_model


def test_build_all_derived_latent_models_saves_fit_and_plot_for_every_entry():
    deck = load_deck()
    names = get_derived_latents(deck)

    results = build_all_derived_latent_models(deck)

    assert set(results.keys()) == set(names.keys())
    for name, paths in results.items():
        assert paths["fit_path"].exists() and paths["fit_path"].stat().st_size > 0, name
        assert paths["plot_path"].exists() and paths["plot_path"].stat().st_size > 0, name
        assert paths["fit_path"].name == "fit.json"
        assert paths["plot_path"].parent.name == "plots"

    print_derived_latent_build_report(results)


def test_build_derived_latent_model_fit_is_a_parametric_ou_model_matching_moment_matched_recombination():
    """The saved fit.json is an effective OU MODEL (same {alpha, beta, i_inf, tau, sig, i0,
    phi_monthly} shape as a founding process's own fit.json), not a table of precomputed monthly
    distributions -- check it against an independently-recomputed moment-matched blend of the three
    components' own fits (see manual_utils.fit_effective_ou)."""
    import json

    from condobuyuq2026.input_deck import (
        _OU_FIT_KEYS,
        get_forecast_model_general,
        get_forecast_model_home_price,
        get_forecast_model_market,
    )
    from condobuyuq2026.utils.manual_utils import fit_effective_ou

    deck = load_deck()
    name = next(iter(get_derived_latents(deck)))
    weights = get_derived_latents(deck)[name]

    paths = build_derived_latent_model(name, deck)
    saved = json.loads(paths["fit_path"].read_text(encoding="utf-8"))

    assert isinstance(saved, dict)
    assert set(saved.keys()) == set(_OU_FIT_KEYS)
    assert 0 < saved["phi_monthly"] < 1

    component_fits = {
        "market": get_forecast_model_market(deck),
        "home_price": get_forecast_model_home_price(deck),
        "general": get_forecast_model_general(deck),
    }
    expected = fit_effective_ou(component_fits, weights)
    for key, value in expected.items():
        assert abs(saved[key] - value) < 1e-9, key


if __name__ == "__main__":
    test_build_all_derived_latent_models_saves_fit_and_plot_for_every_entry()
    test_build_derived_latent_model_fit_is_a_parametric_ou_model_matching_moment_matched_recombination()
    print("All derived_latents checks passed.")
