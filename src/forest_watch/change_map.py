"""Build an interactive Folium map from two aligned, georeferenced forest masks.

Input rasters: one band each; either binary {0,1} or probabilities [0,1].
1 means forest. Newly cleared = forest at date 1 and non-forest at date 2.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.warp import transform_bounds, transform_geom
from shapely.geometry import shape, mapping
from shapely.ops import transform as transform_geometry
from pyproj import Transformer
import folium
from folium.raster_layers import ImageOverlay
from PIL import Image

def rgba(mask, color):
    rgb=np.zeros((*mask.shape,4),dtype=np.uint8); rgb[mask,:3]=color; rgb[mask,3]=175
    return Image.fromarray(rgb)

def build(date1_path,date2_path,out_path,date1_name,date2_name,threshold=.5,max_side=1600,min_polygon_ha=.1):
    with rasterio.open(date1_path) as a, rasterio.open(date2_path) as b:
        if a.crs is None or b.crs is None: raise ValueError('Both date rasters need valid CRS metadata.')
        if (a.crs!=b.crs or a.transform!=b.transform or a.width!=b.width or a.height!=b.height): raise ValueError('Date rasters must share CRS, transform, width, and height; align/resample first.')
        t1=a.read(1).astype(np.float32); t2=b.read(1).astype(np.float32); crs=a.crs; aff=a.transform
        bounds=transform_bounds(crs,'EPSG:4326',*a.bounds,densify_pts=21)
        # Keep browser payload modest while preserving raster display.
        step=max(1,int(np.ceil(max(a.height,a.width)/max_side)))
        f1=(t1[::step,::step]>=threshold); f2=(t2[::step,::step]>=threshold)
        cleared=f1 & ~f2
        mask_transform=aff * rasterio.Affine.scale(step,step)
        raster_crs=crs
    if not f1.any(): raise ValueError('Date 1 mask has no forest pixels under the selected threshold.')
    center=[(bounds[1]+bounds[3])/2,(bounds[0]+bounds[2])/2]
    m=folium.Map(location=center,zoom_start=10,tiles='CartoDB positron',control_scale=True)
    south,west,north,east=bounds; overlay_bounds=[[south,west],[north,east]]
    ImageOverlay(rgba(f1,(31,132,75)),overlay_bounds,name=f'Forest — {date1_name}',opacity=.72,interactive=True,cross_origin=False,zindex=2).add_to(m)
    ImageOverlay(rgba(f2,(44,110,200)),overlay_bounds,name=f'Forest — {date2_name}',opacity=.68,interactive=True,cross_origin=False,zindex=3).add_to(m)
    ImageOverlay(rgba(cleared,(225,35,35)),overlay_bounds,name='Newly cleared (date 1 forest → date 2 non-forest)',opacity=.88,interactive=True,cross_origin=False,zindex=5).add_to(m)
    transformer=Transformer.from_crs(raster_crs,'EPSG:6933',always_xy=True)
    features=[]; min_area=min_polygon_ha*10000
    # Polygonize native-resolution change mask, not display downsample.
    with rasterio.open(date1_path) as a, rasterio.open(date2_path) as b:
        raw1=a.read(1); raw2=b.read(1); loss=(raw1>=threshold)&(raw2<threshold)
        for geom,val in shapes(loss.astype(np.uint8),mask=loss,transform=a.transform):
            if not val: continue
            g=shape(geom); area=float(transform_geometry(transformer.transform,g).area)
            if area<min_area: continue
            wgs=transform_geom(a.crs,'EPSG:4326',geom,precision=6)
            features.append({'type':'Feature','geometry':wgs,'properties':{'area_ha':round(area/10000,3),'from':date1_name,'to':date2_name}})
    total_ha=sum(f['properties']['area_ha'] for f in features)
    geojson={'type':'FeatureCollection','features':features}
    folium.GeoJson(geojson,name=f'Deforestation polygons — {total_ha:,.1f} ha',style_function=lambda _: {'color':'#e32323','weight':2,'fillColor':'#e32323','fillOpacity':.25},tooltip=folium.GeoJsonTooltip(fields=['area_ha','from','to'],aliases=['Area (ha)','From','To'],localize=True)).add_to(m)
    m.fit_bounds(overlay_bounds); folium.LayerControl(collapsed=False).add_to(m)
    out=Path(out_path); out.parent.mkdir(parents=True,exist_ok=True); m.save(out)
    (out.parent/(out.stem+'_deforestation.geojson')).write_text(json.dumps(geojson))
    print(f'map={out} polygons={len(features)} detected_area_ha={total_ha:.3f} display_downsample={step}x')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--date1',required=True); p.add_argument('--date2',required=True); p.add_argument('--date1-name',default='Date 1'); p.add_argument('--date2-name',default='Date 2'); p.add_argument('--threshold',type=float,default=.5); p.add_argument('--min-polygon-ha',type=float,default=.1); p.add_argument('--out',default='outputs/interactive_change_map.html'); a=p.parse_args()
    build(a.date1,a.date2,a.out,a.date1_name,a.date2_name,a.threshold,min_polygon_ha=a.min_polygon_ha)
