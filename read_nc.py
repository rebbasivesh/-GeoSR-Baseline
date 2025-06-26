import xarray as xr
ds = xr.open_dataset("sst.day.mean.2016.nc")
print(ds)
