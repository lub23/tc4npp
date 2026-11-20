"""Monthly exposure map matching the runtime Figure 5 (optional geospatial output)."""
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def plot_monthly_map(monthly_indices, npp_data, figure_dir, province_path=None):
    """Draw the monthly TCIN rose map for coastal NPP sites."""
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    if province_path is None:
        province_path = Path(__file__).resolve().parents[1] / "geo" / "province.geojson"

    npp_monthly_stats = {}
    for npp_name in monthly_indices:
        monthly_means = []
        monthly_stds = []
        for month in range(1, 13):
            if (
                month in monthly_indices[npp_name]
                and monthly_indices[npp_name][month]
            ):
                values = list(monthly_indices[npp_name][month].values())
                monthly_means.append(np.mean(values))
                monthly_stds.append(np.std(values))
            else:
                monthly_means.append(0)
                monthly_stds.append(0)

        npp_monthly_stats[npp_name] = {
            "monthly_means": monthly_means,
            "monthly_stds": monthly_stds,
            "annual_mean": np.mean(monthly_means),
            "annual_std": np.mean(monthly_stds),
            "lat": float(npp_data[npp_name]["Ordinate"]),
            "lon": float(npp_data[npp_name]["Abscissa"]),
            "name": npp_data[npp_name]["en_name"],
        }

    fig = plt.figure(figsize=(12, 10))

    proj = ccrs.PlateCarree(central_longitude=105.0)
    ax = plt.axes([0.05, 0.05, 0.88, 0.90], projection=proj)
    ax.set_extent([98, 125, 15, 45], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.OCEAN, color="#E6F3FF", alpha=0.5)
    ax.add_feature(cfeature.LAND, color="#F5F5DC", alpha=0.3)
    provinces = gpd.read_file(province_path, encoding="utf-8")
    provinces.plot(
        ax=ax,
        facecolor="none",
        edgecolor="gray",
        linewidth=0.8,
        transform=ccrs.PlateCarree(),
    )
    all_means = []
    all_stds = []
    for stats in npp_monthly_stats.values():
        all_means.extend(stats["monthly_means"])
        all_stds.extend(stats["monthly_stds"])

    mean_min, mean_max = 0, np.percentile(all_means, 95) if all_means else 1
    std_min, std_max = 0, np.percentile(all_stds, 95) if all_stds else 1

    cmap = plt.cm.YlOrRd
    norm = plt.Normalize(vmin=mean_min, vmax=mean_max)
    cbar_label = "Mean monthly TCIN"

    base_radius = 0.4
    sorted_npps = [
        (name, npp_monthly_stats[name])
        for name in npp_data.keys()
        if name in npp_monthly_stats
    ]

    offset_list_circle = [
        (-1, 0.5),
        (1, -1),
        (1, 0),
        (-1, 0),
        (1, -0.5),
        (-1.5, 1),
        (1, 2),
        (2, 0.8),
        (-1.5, 1),
        (2.3, -1),
        (1.3, -0.2),
        (1.5, -1.5),
        (0, 2),
        (0, -1.5),
        (-3.7, 3.7),
        (-0.5, -1.6),
        (-1, 1),
        (-2.2, 2),
        (-1.5, 0),
        (-1.5, -0.1),
    ]
    offset_list_name = [
        (-0.5, 0.5),
        (-5, -0.4),
        (-3, 0.8),
        (-3, -0.8),
        (-0.8, 0.25),
        (-4, 2),
        (-1, 0.6),
        (0.7, -0.7),
        (-4, 2.2),
        (0.7, 0.2),
        (1, -1.5),
        (2, -2.5),
        (-2.8, 3.5),
        (-1, -3),
        (-6, 5),
        (-3, -3),
        (-4, 2.5),
        (-8, 2),
        (-10, 0),
        (-5, -1.5),
    ]

    for idx, (npp_name, stats) in enumerate(sorted_npps):
        original_lon = stats["lon"]
        original_lat = stats["lat"]
        circle_offset_lon, circle_offset_lat = (
            offset_list_circle[idx] if idx < len(offset_list_circle) else (1, 1)
        )
        circle_lon = original_lon + circle_offset_lon
        circle_lat = original_lat + circle_offset_lat

        circle_point = ax.projection.transform_point(
            circle_lon, circle_lat, ccrs.PlateCarree()
        )

        size_factor = 1 + (stats["annual_mean"] / (mean_max + 0.001))
        current_base_radius = base_radius * size_factor

        for month_idx in range(12):
            angle_start = 90 - (month_idx + 1) * 30
            angle_end = 90 - month_idx * 30

            month_mean = stats["monthly_means"][month_idx]
            month_std = stats["monthly_stds"][month_idx]

            std_factor = 1 + (month_std / (std_max + 0.001))
            month_radius = current_base_radius * std_factor
            color = cmap(norm(month_mean))

            wedge = Wedge(
                circle_point,
                month_radius,
                angle_start,
                angle_end,
                facecolor=color,
                edgecolor="black",
                linewidth=0.3,
                alpha=0.85,
                transform=ax.transData,
                zorder=15,
            )
            ax.add_patch(wedge)

        ax.plot(
            [original_lon, circle_lon],
            [original_lat, circle_lat],
            "k-",
            linewidth=0.5,
            alpha=0.5,
            transform=ccrs.PlateCarree(),
        )
        ax.plot(
            original_lon,
            original_lat,
            "ro",
            markersize=4,
            transform=ccrs.PlateCarree(),
            zorder=15,
        )
        ax.text(
            original_lon + offset_list_name[idx][0],
            original_lat + offset_list_name[idx][1],
            f"{idx + 1}. {npp_data[npp_name]['en_name']}",
            transform=ccrs.PlateCarree(),
            fontsize=13,
            ha="left",
            va="center",
            color="darkred",
            zorder=16,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.2),
        )

    cax = fig.add_axes([0.82, 0.1, 0.015, 0.8])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label(cbar_label, fontsize=plt.rcParams["axes.labelsize"])

    ax_legend = ax.inset_axes([0.03, 0.04, 0.15, 0.15], projection="polar")

    ax_legend.set_theta_zero_location("N")
    ax_legend.set_theta_direction(-1)

    month_letters = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

    for i in range(12):
        theta_start = np.radians(i * 30 - 15)
        theta_end = np.radians((i + 1) * 30 - 15)
        ax_legend.bar(
            theta_start,
            0.9,
            width=np.radians(30),
            bottom=0,
            color="lightgray",
            alpha=0.5,
            edgecolor="black",
            linewidth=0.3,
        )

        theta_center = theta_start + np.radians(30)
        ax_legend.text(
            theta_center,
            0.75,
            month_letters[i],
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
        )

    ax_legend.set_ylim(0, 1)
    ax_legend.set_yticklabels([])
    ax_legend.set_xticklabels([])
    ax_legend.grid(False)
    ax_legend.patch.set_alpha(0.8)

    radius_meaning = "Radius: Variability"
    ax.text(
        0.02,
        0.01,
        radius_meaning,
        transform=ax.transAxes,
        fontsize=15,
        ha="left",
        va="bottom",
    )

    sub_ax = ax.inset_axes([0.85, 0.045, 0.15, 0.12], projection=proj)
    sub_ax.set_extent([105, 122, 2, 25], crs=ccrs.PlateCarree())

    sub_ax.add_feature(cfeature.OCEAN, color="#E6F3FF", alpha=0.5)
    sub_ax.add_feature(cfeature.LAND, color="#F5F5DC", alpha=0.3)

    provinces.plot(
        ax=sub_ax,
        facecolor="none",
        edgecolor="gray",
        linewidth=0.6,
        transform=ccrs.PlateCarree(),
    )

    for spine in sub_ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)

    ax.text(
        0.86,
        0.01,
        "GS (2024) 0650",
        transform=ax.transAxes,
        fontsize=plt.rcParams["legend.fontsize"],
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, pad=0.3),
    )
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.3, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            os.path.join(str(figure_dir), f"plot_risk_distribution_map.{extension}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)
    return npp_monthly_stats
