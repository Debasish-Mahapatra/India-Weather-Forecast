from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from ecmwf.opendata import Client

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
SHAPEFILE_PATH = BASE_DIR / "Admin2.shp"
LAT_MAX, LAT_MIN = 38.0, 6.0
LON_MIN, LON_MAX = 68.0, 98.0
DEFAULT_STEPS = [6, 12, 120, 240]
DEFAULT_MODEL_ORDER = ["ifs", "aifs-single"]
DEFAULT_VARIABLE_ORDER = ["2t", "tp", "mucape"]

VARIABLES = {
    "2t": {
        "name": "Temperature",
        "cmap": "coolwarm",
        "unit": "°C",
        "data_key": "t2m",
    },
    "tp": {
        "name": "Total Precipitation",
        "cmap": "YlGnBu",
        "unit": "mm",
        "data_key": "tp",
    },
    "mucape": {
        "name": "MUCAPE (Storm Risk)",
        "cmap": "inferno",
        "unit": "J/kg",
        "data_key": "mucape",
    },
}

MODEL_CONFIGS = {
    "ifs": {
        "display_name": "IFS",
        "client_model": "ifs",
        "file_prefix": "",
        "variables": ["2t", "tp", "mucape"],
    },
    "aifs-single": {
        "display_name": "AIFS Single",
        "client_model": "aifs-single",
        "file_prefix": "aifs_single",
        "variables": ["2t", "tp"],
    },
}


def parse_csv_env(name: str) -> list[str] | None:
    raw_value = os.getenv(name)
    if not raw_value:
        return None
    values = [value.strip() for value in raw_value.split(",") if value.strip()]
    return values or None


def get_selected_models() -> list[str]:
    requested_models = parse_csv_env("FORECAST_MODELS") or DEFAULT_MODEL_ORDER
    invalid = [model for model in requested_models if model not in MODEL_CONFIGS]
    if invalid:
        valid = ", ".join(MODEL_CONFIGS)
        raise ValueError(f"Unsupported FORECAST_MODELS entries: {invalid}. Valid values: {valid}")
    return requested_models


def get_selected_steps() -> list[int]:
    requested_steps = parse_csv_env("FORECAST_STEPS")
    if not requested_steps:
        return DEFAULT_STEPS

    try:
        return [int(step) for step in requested_steps]
    except ValueError as exc:
        raise ValueError("FORECAST_STEPS must contain comma-separated integers.") from exc


def get_requested_variables(selected_models: list[str]) -> list[str]:
    supported_variables = {
        variable
        for model in selected_models
        for variable in MODEL_CONFIGS[model]["variables"]
    }

    requested_variables = parse_csv_env("FORECAST_VARIABLES") or DEFAULT_VARIABLE_ORDER
    invalid = [variable for variable in requested_variables if variable not in VARIABLES]
    if invalid:
        valid = ", ".join(VARIABLES)
        raise ValueError(
            f"Unsupported FORECAST_VARIABLES entries: {invalid}. Valid values: {valid}"
        )

    unavailable = [variable for variable in requested_variables if variable not in supported_variables]
    if unavailable:
        models = ", ".join(selected_models)
        raise ValueError(
            f"Requested variables are unavailable for the selected models ({models}): {unavailable}"
        )

    return requested_variables


def get_step_delay_seconds() -> float:
    raw_value = os.getenv("FORECAST_DELAY_SECONDS", "2")
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError("FORECAST_DELAY_SECONDS must be numeric.") from exc


def load_india_map() -> gpd.GeoDataFrame:
    try:
        india_map = gpd.read_file(SHAPEFILE_PATH)
    except Exception as exc:
        raise RuntimeError(
            "Could not load the India shapefile. Ensure the .shp, .shx, .dbf, and .prj files are present."
        ) from exc

    print(f"✅ Shapefile '{SHAPEFILE_PATH.name}' loaded successfully.")
    return india_map


def ensure_2d_field(data_array: xr.DataArray) -> xr.DataArray:
    data = data_array.sel(
        latitude=slice(LAT_MAX, LAT_MIN),
        longitude=slice(LON_MIN, LON_MAX),
    ).squeeze()

    other_dims = [dim for dim in data.dims if dim not in {"latitude", "longitude"}]
    if other_dims:
        data = data.isel({dim: 0 for dim in other_dims})

    if tuple(data.dims) != ("latitude", "longitude"):
        raise ValueError(f"Expected a 2D lat/lon field, got dimensions {data.dims}")

    return data


def convert_units(var_code: str, data: xr.DataArray) -> xr.DataArray:
    units = data.attrs.get("units") or data.attrs.get("GRIB_units")

    if var_code == "2t":
        return data - 273.15
    if var_code == "tp" and units == "m":
        return data * 1000.0
    return data


def format_forecast_time(run_datetime: np.datetime64, step: int) -> str:
    forecast_time = np.datetime64(run_datetime) + np.timedelta64(step, "h")
    return np.datetime_as_string(forecast_time, unit="h").replace("T", " ")


def build_output_path(model_name: str, var_code: str, step: int) -> Path:
    file_prefix = MODEL_CONFIGS[model_name]["file_prefix"]
    if file_prefix:
        filename = f"plot_{file_prefix}_{var_code}_{step}.png"
    else:
        filename = f"plot_{var_code}_{step}.png"
    return BASE_DIR / filename


def render_plot(
    data: xr.DataArray,
    india_map: gpd.GeoDataFrame,
    title: str,
    cmap: str,
    unit: str,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(10, 8))
    try:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        india_map.boundary.plot(ax=ax, color="black", linewidth=1.5)
        data.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            cbar_kwargs={"label": unit},
        )
        plt.title(title, fontsize=14, fontweight="bold")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)


def generate_plot(
    client: Client,
    india_map: gpd.GeoDataFrame,
    model_name: str,
    var_code: str,
    config: dict[str, str],
    step: int,
) -> Path:
    output_path = build_output_path(model_name, var_code, step)

    with TemporaryDirectory(prefix=f"{MODEL_CONFIGS[model_name]['file_prefix'] or model_name}_{var_code}_{step}_") as tmp_dir:
        grib_path = Path(tmp_dir) / f"{var_code}_{step}.grib2"
        result = client.retrieve(
            type="fc",
            step=step,
            param=var_code,
            target=str(grib_path),
        )

        with xr.open_dataset(grib_path, engine="cfgrib") as ds:
            if config["data_key"] not in ds:
                available_keys = ", ".join(sorted(ds.data_vars))
                raise KeyError(
                    f"Missing '{config['data_key']}' in dataset. Available variables: {available_keys}"
                )

            time_str = format_forecast_time(np.datetime64(result.datetime), step)
            data = ensure_2d_field(ds[config["data_key"]])
            data = convert_units(var_code, data)

        render_plot(
            data=data,
            india_map=india_map,
            title=f"{MODEL_CONFIGS[model_name]['display_name']} {config['name']}\n{time_str}",
            cmap=config["cmap"],
            unit=config["unit"],
            output_path=output_path,
        )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Plot was not written correctly: {output_path.name}")

    return output_path


def main() -> int:
    selected_models = get_selected_models()
    selected_steps = get_selected_steps()
    requested_variables = get_requested_variables(selected_models)
    step_delay_seconds = get_step_delay_seconds()
    data_source = os.getenv("ECMWF_OPEN_DATA_SOURCE", "azure")

    run_label = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    print(f"🚀 Starting India Intelligence Pipeline ({run_label})")
    print(f"ℹ️ Data source: {data_source}")
    print(
        "ℹ️ Models: "
        + ", ".join(MODEL_CONFIGS[model]["display_name"] for model in selected_models)
    )

    india_map = load_india_map()
    failures: list[str] = []

    for model_name in selected_models:
        model_config = MODEL_CONFIGS[model_name]
        client = Client(source=data_source, model=model_config["client_model"])
        model_variables = [
            variable for variable in requested_variables if variable in model_config["variables"]
        ]
        skipped_variables = [
            variable for variable in requested_variables if variable not in model_config["variables"]
        ]

        print(f"\nModel: {model_config['display_name']}")
        if skipped_variables:
            print(
                "  ℹ️ Skipping unsupported variables: "
                + ", ".join(skipped_variables)
            )

        for var_code in model_variables:
            config = VARIABLES[var_code]
            print(f"\nProcessing {config['name']}...")

            for step in selected_steps:
                print(f"  - Step {step}h: ", end="", flush=True)
                try:
                    output_path = generate_plot(
                        client=client,
                        india_map=india_map,
                        model_name=model_name,
                        var_code=var_code,
                        config=config,
                        step=step,
                    )
                    print(f"✅ Done ({output_path.name})")
                except Exception as exc:
                    failure_message = (
                        f"{model_config['display_name']} {config['name']} at {step}h failed: {exc}"
                    )
                    failures.append(failure_message)
                    print(f"❌ FAILED: {exc}")

                time.sleep(step_delay_seconds)

    if failures:
        print("\n❌ Pipeline completed with failures:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\n🏁 All requested model plots completed. Dashboard images are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
