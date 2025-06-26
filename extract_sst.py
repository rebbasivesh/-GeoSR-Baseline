import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

# === Load NetCDF ===
ds = xr.open_dataset("sst.day.mean.2016.nc")

# === Set Parameters ===
var_name = "sst"
target_time = "2016-8-11"
lat_min, lat_max = 5, 35
lon_min, lon_max = 45, 105

# === Extract the SST Data ===
sst = ds[var_name].sel(
    time=target_time,
    lat=slice(lat_min, lat_max),
    lon=slice(lon_min, lon_max)
)

# === Plot ===
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

# Set exact region
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

# Plot with true SST values (no normalization, no distortion)
plot = sst.plot.pcolormesh(
    ax=ax,
    cmap="coolwarm",                      # Real SST gradients (blue=cold, red=hot)
    transform=ccrs.PlateCarree(),
    robust=True,
    cbar_kwargs={
        'label': 'SST (°C)',
        'shrink': 0.75
    }
)

# Add map features
ax.coastlines(resolution='10m', color='black')
ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
ax.set_title(f"Sea Surface Temperature (SST)\n{target_time} • {lat_min}°–{lat_max}°N, {lon_min}°–{lon_max}°E")

# === Save ===
os.makedirs("plots", exist_ok=True)
output_path = f"plots/sst_{target_time}_region.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Scientifically accurate SST map saved at: {output_path}")
