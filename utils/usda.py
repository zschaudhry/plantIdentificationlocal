import requests
import pandas as pd

def query_invasive_species_database(scientific_name: str):
    url = "https://apps.fs.usda.gov/arcx/rest/services/EDW/EDW_InvasiveSpecies_01/MapServer/0/query"
    out_fields = [
        "NRCS_PLANT_CODE", "SCIENTIFIC_NAME", "COMMON_NAME", "PROJECT_CODE", "PLANT_STATUS",
        "FS_UNIT_NAME", "EXAMINERS", "LAST_UPDATE"
    ]
    params = {
        'where': f"SCIENTIFIC_NAME='{scientific_name}'",
        'outFields': ",".join(out_fields),
        'returnGeometry': 'true',
        'f': 'json'
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    return None

def format_usda_dates(df: pd.DataFrame) -> pd.DataFrame:
    invasive_df = df.copy()
    for col in invasive_df.columns:
        col_dtype = invasive_df[col].dtype
        if pd.api.types.is_object_dtype(col_dtype):
            sample = invasive_df[col].dropna().astype(str).head(10)
            if sample.str.match(r"^\d{4}-\d{2}-\d{2}T").any() or sample.str.match(r"^\d{4}-\d{2}-\d{2}$").any():
                invasive_df[col] = pd.to_datetime(invasive_df[col], errors='coerce').dt.strftime('%Y-%m-%d')
            elif sample.str.match(r"^\d{12,}").any() or sample.str.match(r"^\d{10,}").any():
                dt = pd.to_datetime(invasive_df[col], errors='coerce', unit='ms')
                if dt.isna().all():
                    dt = pd.to_datetime(invasive_df[col], errors='coerce', unit='s')
                invasive_df[col] = dt.dt.strftime('%Y-%m-%d')
        elif pd.api.types.is_integer_dtype(col_dtype) or pd.api.types.is_float_dtype(col_dtype):
            sample = invasive_df[col].dropna().astype(str).head(10)
            if sample.str.match(r"^\d{12,}").any() or sample.str.match(r"^\d{10,}").any():
                dt = pd.to_datetime(invasive_df[col], errors='coerce', unit='ms')
                if dt.isna().all():
                    dt = pd.to_datetime(invasive_df[col], errors='coerce', unit='s')
                invasive_df[col] = dt.dt.strftime('%Y-%m-%d')
    if 'LAST_UPDATE' in invasive_df.columns:
        try:
            invasive_df['LAST_UPDATE_sort'] = pd.to_datetime(invasive_df['LAST_UPDATE'], errors='coerce')
            invasive_df = invasive_df.sort_values('LAST_UPDATE_sort', ascending=False).drop(columns=['LAST_UPDATE_sort'])
        except Exception:
            pass
    return invasive_df

def extract_usda_coordinates(usda_data: dict) -> pd.DataFrame:
    coords = []
    if usda_data and 'features' in usda_data:
        for f in usda_data['features']:
            geom = f.get('geometry', {})
            if 'x' in geom and 'y' in geom:
                coords.append({'lat': geom['y'], 'lon': geom['x']})
            elif 'points' in geom and isinstance(geom['points'], list):
                for pt in geom['points']:
                    if len(pt) == 2:
                        coords.append({'lat': pt[1], 'lon': pt[0]})
            elif 'paths' in geom and isinstance(geom['paths'], list):
                for path in geom['paths']:
                    for pt in path:
                        if len(pt) == 2:
                            coords.append({'lat': pt[1], 'lon': pt[0]})
            elif 'rings' in geom and isinstance(geom['rings'], list):
                for ring in geom['rings']:
                    for pt in ring:
                        if len(pt) == 2:
                            coords.append({'lat': pt[1], 'lon': pt[0]})
    return pd.DataFrame(coords)
